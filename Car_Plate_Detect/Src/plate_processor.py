import cv2
import numpy as np
import pytesseract
import re
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlateDetector:
    def __init__(self, model_path: str, tesseract_path: str):
        """Initialize the plate detector with model and tesseract paths"""
        self.tesseract_path = tesseract_path
        self.model = None
        self.load_model(model_path)
        self.setup_tesseract()
    
    def setup_tesseract(self):
        """Setup Tesseract OCR path"""
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        logger.info("Tesseract path configured")
    
    def load_model(self, model_path: str):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def preprocess_plate_image(self, plate_img):
        """Enhanced preprocessing for better OCR accuracy"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        filtered = cv2.bilateralFilter(blurred, 11, 17, 17)
        
        height, width = filtered.shape
        scale_factor = max(2, 300 // height)
        resized = cv2.resize(filtered, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Try multiple thresholding methods
        thresh1 = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh2 = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        _, thresh3 = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return [thresh1, thresh2, thresh3]
    
    def extract_text_multiple_configs(self, image):
        """Try multiple OCR configurations"""
        configs = [
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
        
        results = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config).strip()
                results.append(text)
            except Exception as e:
                logger.warning(f"OCR config failed: {e}")
                results.append("")
        
        return results
    
    def clean_and_validate_text(self, text_list):
        """Clean and validate extracted text"""
        cleaned_results = []
        for text in text_list:
            if not text:
                continue
            cleaned = re.sub(r'[^A-Z0-9\s]', '', text.upper())
            cleaned = re.sub(r'\s+', '', cleaned).strip()
            if len(cleaned) >= 4:  # Minimum plate length
                cleaned_results.append(cleaned)
        return cleaned_results
    
    def detect_plate_patterns(self, text_list):
        """Detect various license plate patterns"""
        patterns = [
            r'[A-Z]{2}\d{2}[A-Z]{3}',  # UK format
            r'[A-Z]{3}\d{4}',          # US format
            r'\d{3}[A-Z]{3}',          # Reverse US
            r'[A-Z0-9]{5,8}',          # General 5-8 chars
        ]
        
        matches = []
        for text in text_list:
            for pattern in patterns:
                found = re.findall(pattern, text)
                matches.extend(found)
        
        return matches
    
    def score_plate_candidate(self, candidate):
        """Score plate candidates"""
        if not candidate or len(candidate) < 4:
            return 0
        
        score = 0
        
        # Length scoring
        if 5 <= len(candidate) <= 8:
            score += 10
        elif 4 <= len(candidate) <= 9:
            score += 5
        
        # Check for letters and numbers
        has_letters = bool(re.search(r'[A-Z]', candidate))
        has_numbers = bool(re.search(r'\d', candidate))
        
        if has_letters and has_numbers:
            score += 15
        elif has_letters or has_numbers:
            score += 5
        
        # Pattern bonuses
        if re.match(r'[A-Z]{2}\d{2}[A-Z]{3}', candidate):  # UK format
            score += 20
        elif re.match(r'[A-Z]{3}\d{4}', candidate):  # US format
            score += 15
        
        return score

    def correct_common_ocr_errors(self, plate: str) -> str:
        """Apply common OCR misrecognition corrections based on expected plate formats."""
        if not plate:
            return plate

        corrected = list(plate)

        # Positions 2 and 3 should be digits in UK format: AA00AAA
        if len(plate) >= 4:
            for idx in [2, 3]:
                if corrected[idx] == 'G':
                    corrected[idx] = '6'
                elif corrected[idx] == 'O':
                    corrected[idx] = '0'
                elif corrected[idx] == 'I':
                    corrected[idx] = '1'

        return ''.join(corrected)

    def detect_plate_number(self, image):
        """Main function to detect plate number from image"""
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            results = self.model(image)
            
            best_plate = None
            best_score = 0
            
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    
                    padding = 5
                    xmin = max(0, xmin - padding)
                    ymin = max(0, ymin - padding)
                    xmax = min(image.shape[1], xmax + padding)
                    ymax = min(image.shape[0], ymax + padding)
                    
                    plate_crop = image[ymin:ymax, xmin:xmax]
                    
                    if plate_crop.size == 0:
                        continue
                    
                    thresh_images = self.preprocess_plate_image(plate_crop)
                    all_candidates = []
                    
                    for thresh in thresh_images:
                        text_results = self.extract_text_multiple_configs(thresh)
                        cleaned_texts = self.clean_and_validate_text(text_results)
                        patterns = self.detect_plate_patterns(cleaned_texts)
                        all_candidates.extend(cleaned_texts + patterns)
                    
                    all_candidates = list(set([c for c in all_candidates if c and len(c) >= 4]))
                    
                    for candidate in all_candidates:
                        score = self.score_plate_candidate(candidate)
                        if score > best_score:
                            best_score = score
                            best_plate = candidate
            
            if best_plate:
                best_plate = self.correct_common_ocr_errors(best_plate)
            
            logger.info(f"Best plate detected (after correction): {best_plate} (score: {best_score})")
            return best_plate
        
        except Exception as e:
            logger.error(f"Error in plate detection: {e}")
            return None

    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
