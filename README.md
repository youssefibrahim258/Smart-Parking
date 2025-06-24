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
## Tesseract Installation

This project uses Tesseract OCR. The Tesseract executable is already included in the `tesseract` folder.

### If you move the project to another machine:
- Make sure the `tesseract` folder is kept in the project root.
- Make sure `tesseract.exe` and the `tessdata` folder are inside it.

The path is already configured in `config.py`.

You do not need to install Tesseract globally.




## Contact

**AI Engineer:** [mohamed-ehab415](https://github.com/mohamed-ehab415)
