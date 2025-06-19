#  Smart Parking

<p align="center">
  <img src="https://github.com/youssefibrahim258/Smart-Parking/blob/master/logo%20parking.jpg?raw=true" alt="Smart Parking Logo" width="70%" height="400"/>
</p>


Smart Parking is an intelligent parking management system designed to streamline the search for parking in crowded areas. Leveraging machine learning and real-time camera input, Smart Parking detects car plates at entry gates, automates vehicle registration, and enables users to locate and reserve available spaces through a dedicated mobile application.

---

##  Features

- **Car Plate Detection**  
  Real-time license plate recognition utilizing YOLO and Python for accurate and fast detection.

- **Automatic Gate Access Control**  
  Seamless entry and exit automation based on vehicle registration and plate recognition.

- **Mobile App (Flutter)**  
  Cross-platform mobile application provides users with real-time parking space availability.

- **Store Selection & Navigation**  
  Recommends the nearest parking spot based on the user's selected destination, ensuring convenience.

- **Reservation System**  
  Allows users to reserve parking spots in advance, minimizing wait times and improving user experience.

---

## Technologies Used

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" height="40"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" height="40"/>
  <img src="https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white" alt="Flutter" height="40"/>
  <img src="https://img.shields.io/badge/YOLO-FFBB00?style=for-the-badge" alt="YOLO" height="40"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=probot&logoColor=white" alt="Machine Learning" height="40"/>
</p>

- **Python** – Backend logic and image processing  
- **FastAPI** – High-performance API development  
- **Flutter** – Cross-platform mobile application  
- **YOLO** – Real-time car plate detection (You Only Look Once)  
- **Machine Learning** – Object detection and system intelligence

________________________________________________________________________

# Car Plate Detection Model

## Overview

Welcome to the **Car Plate Detection Model** !  
. The system leverages YOLOv11 for car plate detection and integrates Optical Character Recognition (OCR) for precise plate number extraction, enabling automated gate access and vehicle registration.

---

## Features

- **AI-Powered Plate Detection:**  
  Utilizes YOLOv11 for robust and accurate car plate localization.

- **High-Quality Dataset:**  
  Trained and validated on the Roboflow Plate Dataset, comprising 2,600 expertly annotated images.

- **Accurate Plate Number Extraction:**  
  Employs advanced OCR techniques to read plate numbers from detected regions with high precision.

- **REST API Deployment:**  
  FastAPI-based backend serves real-time detection and recognition results via RESTful endpoints.

- **Smart Parking Integration:**  
  Seamless integration with parking management systems for automatic gate control and vehicle registration.

- **Performance Monitoring:**  
  Comprehensive evaluation using confusion matrices, accuracy metrics, and visualization tools.

- **Continuous Improvement:**  
  Detection accuracy enhanced through iterative model fine-tuning and OCR post-processing.

- **Version Control and Documentation:**  
  All development, experiments, and results are tracked and managed using GitHub.

---

## Architecture

```mermaid
graph TD
  A[Input Image/Video] --> B[YOLOv11 Plate Detection]
  B --> C[OCR Plate Number Extraction]
  C --> D["REST API (FastAPI)"]
  D --> E["Smart Parking System"]
  E --> F[Gate Control & Vehicle Registration]
```

---

## Dataset

- **Source:** Roboflow Plate Dataset  
- **Size:** 2,600 annotated images  
- **Preparation:** Downloaded, cleaned, and split for training and validation.

---

## Results


### Results 

<img src="https://github.com/youssefibrahim258/Smart-Parking/blob/master/Car_Plate_Detect/outputs/results.png?raw=true" width="600"/>

### Confusion Matrix

<img src="https://github.com/youssefibrahim258/Smart-Parking/blob/master/Car_Plate_Detect/outputs/confusion_matrix_normalized.png?raw=true" width="600"/>


### Results Summary

- **Detection Accuracy:** High precision and recall for plate detection.
- **OCR Performance:** Minimized misreadings through post-processing and fine-tuning.
- **System Throughput:** Real-time performance suitable for live parking environments.

---



## Contact

**AI Engineer:** [mohamed-ehab415](https://github.com/mohamed-ehab415)


---
---




# Parking Slot Detection System

This project detects whether parking spots are **empty** or **not empty** using a trained Support Vector Machine (SVM) classifier. It includes:

- A script to train the SVM model.
- A real-time video detection script.
- A FastAPI backend to serve predictions from the trained model.

---

##  Project Structure

```
ParkingDetector/
│
├── train_svm.py             # Train SVM classifier on image data
├── main.py                  # Process video to detect parking status
├── util.py                  # Helper functions (classification + cropping)
├── api.py                   # FastAPI API for external access
├── SVM_model                # Trained model saved as a pickle file
├── mask_1920_1080.png       # Binary mask image for parking spots
└── requirements.txt         # Python dependencies
```

---

##  How the Model Works

- Images of parking spots are resized to `15x15` pixels.
- Each image is flattened into a feature vector and fed into an SVM classifier.
- The model is trained on labeled data:
  - `0` → empty
  - `1` → not empty
- `GridSearchCV` is used for hyperparameter tuning.

---

## ▶ How to Run the Project

### 1. Train the Model (Optional - model already provided)

```bash
python train_svm.py
```

### 2. Run the Video Detection App

Make sure you have:
- `mask_1920_1080.png` → the parking area mask image
- A video file path defined in `main.py`

Then run:

```bash
python main.py
```

This will display:
- Parking spots outlined:
  - **Green** = empty
  - **Red** = not empty
- Region labels (A, B, C, D)
- Live count of empty slots on the video

---

##  Run the FastAPI Backend

### Step 1: Install requirements

```bash
pip install -r requirements.txt
```

### Step 2: Start the FastAPI server

```bash
uvicorn api:app --reload
```

---

## API Usage

### `POST /status`

Send an image frame and receive a JSON response with parking availability per region.

#### Request Details:
- **Method**: `POST`
- **Endpoint**: `/status`
- **Body Type**: `multipart/form-data`
- **Field**: `file` → image frame (JPG/PNG)

####  Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/status" -F "file=@frame.jpg"
```

####  Sample Response:

```json
{
  "a": { "empty": 20, "total": 100 },
  "b": { "empty": 15, "total": 120 },
  "c": { "empty": 3, "total": 80 },
  "d": { "empty": 55, "total": 90 }
}
```


##  Dependencies

Make sure you have the following Python libraries installed:

```txt
fastapi
uvicorn
opencv-python
scikit-learn
scikit-image
numpy
```

You can install them via:

```bash
pip install -r requirements.txt
```

---

##  Notes

- The `mask_1920_1080.png` file defines the location of each parking spot using connected components.
- Each parking spot is evaluated individually using the trained SVM classifier.
- You can adjust the region splitting logic or video input path in `main.py`.
- The backend API can be used by any client (mobile app, web app, etc.) to check real-time parking availability by sending frames.

---

##  Author

Developed by **Yousef Ibrahim** —(https://github.com/youssefibrahim258).

