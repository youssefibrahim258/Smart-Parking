from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np

# Load model
model = YOLO(r"E:\car_plate_detect\models\best.pt")

# Load image
img_path = r"E:\car_plate_detect\Dataset\why-are-number-plates-yellow-and-white.jpg"
img = cv2.imread(img_path)

# Run inference
results = model(img)

# Process first image result
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Get bounding box coordinates
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        # Crop plate region
        plate_crop = img[ymin:ymax, xmin:xmax]

        # Preprocess for OCR
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        text = pytesseract.image_to_string(thresh, config='--psm 8')
        print("Detected Plate Number:", text.strip())
