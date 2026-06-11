# MBG Freshness AI — YOLOv8 Object Detection

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![YOLOv8 Version](https://img.shields.io/badge/ultralytics-v8.0+-red.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch Version](https://img.shields.io/badge/pytorch-v2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask Version](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)

An AI-powered real-time freshness object detection system tailored for the Makan Bergizi Gratis (MBG) national program. Built using Ultralytics YOLOv8 Object Detection and served via a lightweight, high-performance Flask backend with a premium, responsive glassmorphic dark interface.

Developed by Kelompok 11 — Politeknik Caltex Riau:
* Ghaswul Fikri Fadhillah (<ghaswul23ti@mahasiswa.pcr.ac.id>)
* Daffa Hadziq (<daffa23ti@mahasiswa.pcr.ac.id>)
* Dafi Hibrizi (<dafi23ti@mahasiswa.pcr.ac.id>)

---

## System Architecture & Workflow

Here is how the components interact in the MBG Freshness AI application:

```mermaid
graph TD
    A[Client Browser] -->|1. Stream Camera / Upload Image| B(Webcam/Upload Module)
    B -->|2. Capture Frame & Convert to Base64| C[JS Controller]
    C -->|3. POST Request with Base64 Payload| D(Flask Backend Server)
    D -->|4. Decode Base64 to cv2 Image Matrix| E[Image Processor]
    E -->|5. Convert BGR to RGB| F[RGB NumPy Array]
    F -->|6. Feed Forward Inference| G[YOLOv8 Detection Model]
    G -->|7. Return Bounding Boxes & Confidence| H[Piecewise Suitability Scorer]
    H -->|8. Compute Suitability % & Recommendation| I[Flask JSON Response]
    I -->|9. AJAX Success Handler| C
    C -->|10. Draw Viewport Boxes & Dynamic Stats| A
```

### System Workflow Description
1. **Frontend Capture:** The browser captures real-time video frames from the webcam or processes uploaded files, rendering the output on a canvas.
2. **Data Transmission:** Every frame is serialized into a Base64 string and sent asynchronously via the Fetch API to the `/predict_frame` backend endpoint.
3. **Backend Processing:** Flask receives the payload, decodes it into a standard OpenCV image matrix (numpy array), and converts the channels from BGR to RGB.
4. **AI Inference:** The trained YOLOv8 Object Detection model (`runs/detect_freshness/weights/best.pt`) predicts bounding boxes and class labels (fresh / rotten) for each ingredient.
5. **Suitability Scoring:** The relative ratio of fresh versus rotten items in the frame is evaluated and mapped to a percentage suitability index using a custom piecewise linear algorithm.
6. **UI Rendering:** The result is returned as JSON to the client, which dynamically renders neon bounding boxes around each detected fruit, updates average session statistics, logs recent items, and updates the dynamic suitability score bar without page reloads.

---

## Key Features

### 1. Real-time Video Inference
* **Webcam Streaming:** Scans raw food ingredients via the device webcam.
* **Media Uploads:** Supports scanning via uploaded static images or pre-recorded videos.
* **Scan Viewport Effects:** Interactive neon scanning lines and color-coded bounding boxes representing prediction status (green for fresh, red for rotten).

### 2. Consumption Suitability Index
* **Piecewise Linear Scoring:** Calculates a precise percentage suitability score based on fruit counts and detection confidence.
* **Actionable UX Recommendations:** Displays smart warnings based on safety thresholds (e.g., highly fresh, consume today, or discard/compost).

### 3. Interactive Floating "Dynamic Island" Navbar
* **Scroll-Responsive Capsule:** Smoothly morphs from a full header down to a compact floating capsule on scroll to maximize viewport estate.
* **Pulse & Glow States:** Border glows green dynamically during active AI inference.
* **Mobile-First Design:** Autocompacts on mobile viewports (< 768px) to fit all screen sizes perfectly.

### 4. Interactive Statistics & History
* **Live Session Metrics:** Displays total items scanned, average object detection confidence, and total count stats of fresh and rotten ingredients.
* **Scan Log:** Interactive time-stamped history list showing the 20 most recent prediction outcomes.

### 5. Training Report Dashboard
* **val_accuracy & val_loss metrics:** Shows key performance results from training logs.
* **Interactive Evaluation Charts:** Embeds training accuracy curves, training loss curves, and the evaluation confusion matrix.
* **Persisted Dark & Light Mode:** Switches themes smoothly with high-contrast neon stabilo highlights that persist across page reloads.

---

## Installation & Setup

### Step 1: Dataset Preparation
1. Download the dataset in YOLOv8 format from Roboflow.
2. Extract the ZIP file and place the directory inside the project root, naming it `yolo_detection_dataset`.
   The expected directory structure is:
   ```text
   yolo_detection_dataset/
   ├── train/
   ├── valid/
   ├── test/
   └── data.yaml
   ```

### Step 2: Model Training
Open the Jupyter Notebook [model_training.ipynb](model_training.ipynb) and execute all cells (Run All).
The notebook will automatically download the pre-trained `yolov8n.pt` base weights, install all required dependencies from `requirements.txt`, train the object detection model for 25 epochs, and export the best-performing weights to `runs/detect_freshness/weights/best.pt`.

### Step 3: Web Application Deployment
Once training is complete and the `best.pt` weights are exported, run the following commands in your terminal to deploy the Flask application:
```bash
cd "mbg_freshness_ai_YOLO"
pip install -r requirements.txt
python web/app.py
```
Open your web browser and navigate to `http://localhost:8080`.
