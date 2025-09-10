# 🚗 Smart Parking System  

<p align="center">
  <img src="https://github.com/youssefibrahim258/Smart-Parking/blob/master/logo%20parking.jpg?raw=true" alt="Smart Parking Logo" width="70%" height="400"/>
</p>

---

## 📖 Introduction  
Smart Parking is an **AI-powered parking management system** designed for shopping malls.  
It detects vacant spaces, recognizes license plates, and provides a mobile app for reservation and navigation.  

### 🔍 Problem  
Finding a parking spot in crowded areas wastes time, increases congestion, and frustrates drivers.  

### 🎯 Solution  
- Real-time parking spot detection  
- Automated vehicle registration via license plate recognition  
- Mobile app for reservation & navigation  

### ✨ Benefits  
- ⏱ Faster parking experience  
- 🅿️ Optimized space utilization  
- 🚦 Reduced congestion  
- 😊 Improved customer satisfaction  

---

## 🏗️ System Architecture  

1. **License Plate Recognition (YOLOv11 + OCR)**  
2. **Parking Spot Detection (SVM + CV)**  
3. **Backend (FastAPI + PostgreSQL)**  
4. **Mobile App (Flutter)**  

<p align="center">
  <img src="assets/image2.png" width="80%" alt="Architecture"/>
</p>

---

## 🔬 Methodology  

### 1️⃣ License Plate Dataset  
- Source: Custom dataset via **Roboflow**  
- Classes: `license_plate`  
- Preprocessing: resize (640×640), normalize, augmentations  

<p align="center">
  <img src="assets/image4.png" width="60%" alt="Dataset Example"/><br>
  📸 <i>Figure 4: Sample labeled images from the license plate dataset.</i>
</p>

### 2️⃣ Parking Slot Detection Dataset  
- Classes: `empty`, `not_empty`  
- Preprocessing: resize to 15×15, flatten, label encoding  

<p align="center">
  <img src="assets/image5.png" width="40%" alt="Parking Spot Example"/><br>
  📸 <i>Figure 5: Sample image of a parking spot.</i>
</p>

---

## ⚙️ Core Functionalities  

- **Plate Recognition** → Detect + OCR plate number  
- **Parking Spot Detection** → Real-time empty/occupied classification  
- **Mobile App** → Reserve, navigate, find car  
- **Fee Estimation** → Auto calculation on exit  
- **Database** → Centralized sessions & history  

---

## 🛠️ Tools & Technologies  

| Tool           | Purpose |
|----------------|---------|
| **YOLOv11**    | License plate detection |
| **OCR + OpenCV** | Plate text extraction |
| **SVM (sklearn)** | Parking spot classification |
| **FastAPI**    | Backend communication |
| **PostgreSQL** | Database management |
| **Flutter**    | Mobile application |
| **Roboflow**   | Dataset annotation |
| **Kaggle API** | Cloud GPU training |

---

## 📊 Results  

### 🔹 License Plate Detection (YOLOv11 + OCR)  
- mAP@0.5: High accuracy  
- Strong performance but sensitive to low lighting & occlusion  

<p align="center">
  <img src="assets/image15.png" width="70%" alt="Training Curves"/><br>
  📸 <i>Figure 15: YOLOv11 training & validation curves.</i>
</p>

### 🔹 Parking Spot Classification (SVM)  
- Accuracy: **100%** (controlled dataset)  
- Limitation: needs more diverse data for generalization  

---

## 🚀 Installation & Usage  

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/Smart-Parking.git
cd Smart-Parking
