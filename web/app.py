import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

app = Flask(__name__)

# Locate the trained YOLO object detection model relative to this app.py file
# (app.py is in 'YOLO VERSION/web/', model is in 'YOLO VERSION/runs/detect_freshness/weights/best.pt')
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(script_dir), 'runs', 'detect_freshness', 'weights', 'best.pt')

try:
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"YOLO Detection Model successfully loaded from: {model_path}")
    else:
        print(f"Warning: YOLO Detection Model weights not found at {model_path}. You will need to complete training first.")
        model = None
except Exception as e:
    print(f"Error loading YOLO model from {model_path}: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({"status": "ready" if model is not None else "error"})

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    global model
    if model is None:
        # Check if the model has been created since server start
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                print(f"YOLO Model loaded dynamically from: {model_path}")
            except Exception as e:
                return jsonify({"success": False, "error": f"Model not loaded and reload failed: {e}"}), 500
        else:
            return jsonify({"success": False, "error": f"Model weights not found at {model_path}. Run training first!"}), 500
        
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data"}), 400
            
        # Decode base64 image data
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB (YOLO handles resizing and normalizing internally)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO Object Detection inference (imgsz=640, matching training size)
        results = model(img_rgb, imgsz=640, verbose=False)
        
        # Parse Object Detection results
        boxes = results[0].boxes
        names = model.names # Dictionary e.g. {0: 'fresh', 1: 'rotten'}
        
        # Determine indexes for fresh and rotten
        # In Roboflow: names are typically 'fresh' and 'rotten'
        # Let's locate them robustly
        fresh_idx = [k for k, v in names.items() if v.lower() == 'fresh']
        rotten_idx = [k for k, v in names.items() if v.lower() == 'rotten']
        
        # Fallback if names are different
        fresh_class_id = fresh_idx[0] if fresh_idx else 0
        rotten_class_id = rotten_idx[0] if rotten_idx else 1
        
        count_fresh = 0
        count_rotten = 0
        boxes_list = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Map classes: fresh -> 'segar', rotten -> 'tidak_segar'
            label_name = "segar" if cls == fresh_class_id else "tidak_segar"
            
            if label_name == "segar":
                count_fresh += 1
            else:
                count_rotten += 1
                
            boxes_list.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf * 100.0,
                "class_id": cls,
                "label": label_name
            })
            
        total = count_fresh + count_rotten
        if total > 0:
            prob_tidak = count_rotten / total
            prob_segar = count_fresh / total
            prediction_val = prob_tidak
        else:
            prob_tidak = 0.0
            prob_segar = 1.0
            prediction_val = 0.0
            
        segar_prob = prob_segar * 100.0
        tidak_prob = prob_tidak * 100.0
        
        label = "segar" if prediction_val < 0.5 else "tidak_segar"
        
        # Logika Kelayakan Piecewise Linear (Ghaswul Fikri Fadhillah)
        if total == 0:
            kelayakan = 100.0
            kelayakan_status = "Tidak ada bahan makanan terdeteksi"
        elif prediction_val < 0.5:
            kelayakan = 100.0 - (prediction_val * 60.0)
            kelayakan_status = "LAYAK KONSUMSI (Sangat Segar & Aman)"
        elif prediction_val < 0.7:
            kelayakan = 70.0 - ((prediction_val - 0.5) * 100.0)
            kelayakan_status = "LAYAK KONSUMSI (Batas Aman - Konsumsi Segera!)"
        else:
            kelayakan = 50.0 - (((prediction_val - 0.7) / 0.3) * 50.0)
            kelayakan_status = "TIDAK LAYAK KONSUMSI (Busuk/Rusak - Jangan Diolah!)"
            
        kelayakan = max(0.0, min(100.0, kelayakan))
        
        return jsonify({
            "success": True,
            "label": label,
            "segar": segar_prob,
            "tidak": tidak_prob,
            "kelayakan": kelayakan,
            "kelayakan_status": kelayakan_status,
            "boxes": boxes_list
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Memulai server web YOLO MBG AI (Self-Contained)...")
    print(f"Looking for model at: {model_path}")
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
