---
title: VisionGuard AI
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
---

# VisionGuard AI Intelligent Traffic Violation Detection System

VisionGuard AI is a comprehensive deep learning pipeline designed to automatically detect and log traffic violations from video feeds. Leveraging fine tuned YOLOv8 models and advanced object tracking, the system can accurately detect various violations.

## Features

* Multi Violation Detection: Detects riders without helmets, seatbelt violations, and extracts License Plates.
* Advanced Processing Pipeline: Uses YOLOv8 for high speed object detection and tracking.
* Evidence Generation: Automatically captures and crops images of violating vehicles and generates comprehensive PDF reports for fines.
* Interactive UI: Built with Gradio for an intuitive web based interface that allows users to process video feeds, view live inference, and download evidence.
* Scalable Architecture: Modular codebase enabling easy addition of new violation types.

## Project Structure

* `app.py`: Main Gradio Web Application
* `core/detection_engine.py`: Inference pipeline
* `core/tracker.py`: Object tracking utilities
* `utils/evidence_generator.py`: PDF and image extraction for violations
* `utils/visualizer.py`: Bounding box and label rendering
* `models/`: Directory for pre trained weights
* `requirements.txt`: Python dependencies

## Instructions to Run

### 1. Clone the repository

```bash
git clone https://github.com/Abhyuday-06/Traffic-Violation-Detection-System.git
cd Traffic-Violation-Detection-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```
*(On Windows use: `venv\Scripts\activate`)*

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

Ensure the following weights are placed in the root or `models` directory:

* `yolov8n.pt`
* `helmet_best.pt`
* `lp_best.pt`
* `seatbelt_best.pt`

### 5. Run the Application

```bash
python app.py
```
The application will launch on your local host (usually `http://127.0.0.1:7860/`). 

### 6. Usage

1. Upload a traffic video via the web interface.
2. Select the violations you wish to detect.
3. Click Process Video.
4. Once completed, you can view the annotated video and download the generated PDF evidence reports.

## Submission Details

This is the submission for Flipkart Gridlock 2.0 Round 2.
