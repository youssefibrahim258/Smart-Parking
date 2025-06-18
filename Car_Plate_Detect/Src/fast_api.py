from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import logging
from pathlib import Path

# Import our custom modules
from plate_processor import PlateDetector
from config import (
    MODEL_PATH, TESSERACT_PATH, API_HOST, API_PORT, 
    API_TITLE, API_DESCRIPTION, API_VERSION,
    CORS_ORIGINS, CORS_CREDENTIALS, CORS_METHODS, CORS_HEADERS,
    LOG_LEVEL, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    validate_paths
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Validate configuration
try:
    validate_paths()
    logger.info("Configuration validated successfully")
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    exit(1)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_CREDENTIALS,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Initialize plate detector
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize the plate detector on startup"""
    global detector
    try:
        detector = PlateDetector(MODEL_PATH, TESSERACT_PATH)
        logger.info("Plate detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize plate detector: {e}")

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    # Check content type
    if not file.content_type or not file.content_type.startswith('image/'):
        return False
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    return True

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "License Plate Detection API", 
        "status": "running",
        "version": API_VERSION
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": detector is not None and detector.is_model_loaded()
    }

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    """
    Upload an image and get the detected license plate number.
    Returns only the plate number as a string.
    """
    
    # Check if detector is initialized
    if detector is None:
        raise HTTPException(status_code=503, detail="Service not ready - detector not initialized")
    
    # Validate file
    if not validate_image_file(file):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file. Must be an image with extension: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Convert to OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        logger.info(f"Processing image: {file.filename}, size: {image.shape}")
        
        # Detect plate number
        plate_number = detector.detect_plate_number(image)
        
        if plate_number:
            logger.info(f"Plate detected: {plate_number}")
            return plate_number  # Return only the string
        else:
            logger.info("No plate detected")
            return "NO_PLATE_DETECTED"
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/info")
async def api_info():
    """Get API information and configuration"""
    return {
        "api_title": API_TITLE,
        "api_version": API_VERSION,
        "model_loaded": detector is not None and detector.is_model_loaded(),
        "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /info": "Detailed API information",
            "POST /detect": "Upload image and detect plate number"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)