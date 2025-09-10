# Smart Parking System 🚗

<p align="center">
  <img src="images/logo_parking.jpg" alt="Smart Parking Logo" width="300"/>
</p>

<p align="center">
  <strong>AI-Powered Parking Management for Shopping Malls</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Flutter-3.0+-blue.svg" alt="Flutter Version"/>
  <img src="https://img.shields.io/badge/FastAPI-0.68+-green.svg" alt="FastAPI Version"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Overview

Smart Parking is an **AI-powered parking management system** designed for shopping malls. It helps drivers save time by automatically detecting vacant spaces, recognizing license plates, and allowing users to reserve and locate their cars through a mobile app.

### Problem Statement
- Finding parking spots in crowded areas wastes time and causes traffic congestion
- Manual parking management is inefficient and costly
- Customers experience delays and dissatisfaction
- Reduced mall sales due to parking difficulties

### Solution
Our system combines **real-time camera data**, **AI models**, and a **mobile application** to automate:
- ✅ Detecting available parking spots using computer vision
- ✅ Vehicle registration via license plate recognition
- ✅ Reservation and navigation through mobile app
- ✅ Automatic fee calculation and payment processing

## Features

### Core Functionality
- **Real-time Space Detection**: AI-powered detection of available parking spots
- **License Plate Recognition**: Automatic vehicle identification using YOLOv11 + OCR
- **Mobile App Integration**: Flutter-based cross-platform application
- **Smart Reservations**: Book parking spots in advance
- **Destination-based Guidance**: Recommend parking near target stores
- **Automatic Fee Calculation**: Duration-based pricing with automated billing
- **Find My Car**: Help users locate their parked vehicles

### User Benefits
- ⏱️ Reduced waiting time
- 📱 Convenient mobile interface
- 🎯 Destination-aware parking suggestions
- 💳 Cashless payment system
- 🔍 Easy car location tracking

## Architecture

<p align="center">
  <img src="images/architecture_diagram.png" alt="System Architecture" width="800"/>
</p>

The Smart Parking system follows a **modular, layered architecture**:

1. **Computer Vision Layer**: YOLOv11 + OCR for license plate recognition, SVM for space detection
2. **Backend Layer**: FastAPI-based REST API with PostgreSQL database
3. **Mobile Layer**: Flutter application for user interaction
4. **Integration Layer**: Real-time synchronization between all components

### Workflow
```
Vehicle Entry → Camera Capture → AI Processing → Database Update → Mobile App Sync
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI | Lightweight REST API |
| **Database** | PostgreSQL | Data storage and management |
| **Mobile App** | Flutter | Cross-platform mobile application |
| **AI Models** | YOLOv11, OpenCV, SVM | Computer vision and ML |
| **Dataset** | Roboflow | Data annotation and preprocessing |
| **Training** | Kaggle API | Cloud-based model training |
| **OCR** | OpenCV + Tesseract | Text extraction from images |

## Installation

### Prerequisites
- Python 3.8+
- Flutter 3.0+
- PostgreSQL 12+
- Git

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/smart-parking.git
cd smart-parking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
createdb smart_parking
python manage.py migrate

# Start the server
uvicorn main:app --reload
```

### Mobile App Setup
```bash
# Navigate to mobile app directory
cd mobile_app

# Install Flutter dependencies
flutter pub get

# Run the app
flutter run
```

### Environment Variables
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://username:password@localhost/smart_parking
SECRET_KEY=your-secret-key
ROBOFLOW_API_KEY=your-roboflow-key
```

## Usage

### For Developers

#### Training the License Plate Model
```python
from models.yolo_trainer import YOLOTrainer

trainer = YOLOTrainer()
trainer.train_model(
    dataset_path="data/license_plates",
    epochs=50,
    batch_size=8
)
```

#### Running Space Detection
```python
from models.parking_detector import ParkingDetector

detector = ParkingDetector()
results = detector.detect_spaces("path/to/parking_image.jpg")
```

### For End Users

1. **Download the mobile app**
2. **Register with your license plate number**
3. **View available parking spots in real-time**
4. **Reserve a spot before arrival**
5. **Navigate to your reserved spot**
6. **Automatic check-in via QR code**
7. **Receive parking location reminder**
8. **Automatic fee calculation on exit**

## API Documentation

### License Plate Recognition
```http
POST /api/v1/detect-plate
Content-Type: multipart/form-data

{
  "image": "base64_encoded_image"
}
```

### Parking Availability
```http
GET /api/v1/parking/availability
```

### Reserve Parking Spot
```http
POST /api/v1/parking/reserve
Content-Type: application/json

{
  "user_id": "123",
  "spot_id": "A01",
  "duration": 120
}
```

For complete API documentation, visit: `http://localhost:8000/docs`

## Model Performance

### License Plate Detection (YOLOv11)
- **Dataset Size**: 2,640 annotated images
- **Training Accuracy**: 94.2%
- **Validation Accuracy**: 91.8%
- **Inference Time**: ~50ms per image

<p align="center">
  <img src="images/yolo_training_curves.png" alt="Training Curves" width="600"/>
</p>

### Parking Space Classification (SVM)
- **Accuracy**: 100% on test set
- **Precision**: 1.0
- **Recall**: 1.0
- **F1-Score**: 1.0

<p align="center">
  <img src="images/confusion_matrix.png" alt="Confusion Matrix" width="400"/>
</p>

## Screenshots

### Mobile Application

<p align="center">
  <img src="images/app_home.png" alt="Home Screen" width="200"/>
  <img src="images/app_availability.png" alt="Parking Availability" width="200"/>
  <img src="images/app_reservation.png" alt="Reservation" width="200"/>
  <img src="images/app_payment.png" alt="Payment" width="200"/>
</p>

### License Plate Recognition

<p align="center">
  <img src="images/plate_detection.png" alt="License Plate Detection" width="400"/>
  <img src="images/ocr_result.png" alt="OCR Result" width="400"/>
</p>

### Parking Space Detection

<p align="center">
  <img src="images/parking_detection.png" alt="Parking Space Detection" width="600"/>
</p>

### Database Schema

<p align="center">
  <img src="images/database_erd.png" alt="Database ERD" width="600"/>
</p>

## File Structure

```
smart-parking/
├── backend/
│   ├── models/
│   │   ├── yolo_model.py
│   │   ├── svm_model.py
│   │   └── ocr_processor.py
│   ├── api/
│   │   ├── routes/
│   │   └── middleware/
│   ├── database/
│   │   ├── models.py
│   │   └── migrations/
│   └── main.py
├── mobile_app/
│   ├── lib/
│   │   ├── screens/
│   │   ├── models/
│   │   └── services/
│   ├── assets/
│   └── pubspec.yaml
├── datasets/
│   ├── license_plates/
│   └── parking_spaces/
├── images/
├── docs/
├── requirements.txt
└── README.md
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use Flutter best practices for mobile development
- Write unit tests for new features
- Update documentation for API changes

## Roadmap

- [ ] Multi-language support for mobile app
- [ ] Integration with payment gateways
- [ ] Advanced analytics dashboard
- [ ] Support for electric vehicle charging stations
- [ ] Integration with mall navigation systems
- [ ] Predictive parking availability using ML

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## Acknowledgments

- Thanks to Roboflow for dataset management tools
- OpenCV community for computer vision libraries
- Flutter team for cross-platform development framework
- FastAPI for the excellent web framework

## Support

If you have any questions or need help, please:
- Open an issue on GitHub
- Email us at support@smartparking.com
- Check our [documentation](docs/)

---

<p align="center">
  Made with ❤️ for smarter cities
</p>
