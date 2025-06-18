import os
from pathlib import Path

# Model Configuration
MODEL_PATH = r"E:\car_plate_detect\models\best.pt"

# Tesseract Configuration
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "License Plate Detection API"
API_DESCRIPTION = "API for detecting license plates and returning plate numbers"
API_VERSION = "1.0.0"

# CORS Configuration
CORS_ORIGINS = ["*"]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Logging Configuration
LOG_LEVEL = "INFO"

# File Upload Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Processing Configuration
MIN_PLATE_LENGTH = 4
MAX_PLATE_LENGTH = 12
CONFIDENCE_THRESHOLD = 0.5

def validate_paths():
    """Validate that required files exist"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    if not os.path.exists(TESSERACT_PATH):
        raise FileNotFoundError(f"Tesseract executable not found: {TESSERACT_PATH}")
    
    return True