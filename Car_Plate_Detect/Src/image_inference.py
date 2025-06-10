from ultralytics import YOLO
import cv2

# Load trained YOLO model (adjust path to your weights)
model = YOLO(r"E:\car_plate_detect\models\best.pt")

# Load image
image = cv2.imread("E:\car_plate_detect\Dataset\OIP.jpeg")

# Run inference
results = model(image)

# Plot results on image
annotated_frame = results[0].plot()

# Display the result
cv2.imshow('YOLO Inference', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
