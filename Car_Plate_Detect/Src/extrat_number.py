from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import re

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_plate_image(plate_img):
    """Enhanced preprocessing for better OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(blurred, 11, 17, 17)
    
    # Resize image for better OCR (scaling up helps with small text)
    height, width = filtered.shape
    scale_factor = max(2, 300 // height)  # Ensure minimum height of 300px
    resized = cv2.resize(filtered, None, fx=scale_factor, fy=scale_factor, 
                        interpolation=cv2.INTER_CUBIC)
    
    # Try multiple thresholding methods and return the best one
    thresh_methods = []
    
    # Method 1: Adaptive thresholding (Gaussian)
    thresh1 = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh_methods.append(thresh1)
    
    # Method 2: Adaptive thresholding (Mean)
    thresh2 = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh_methods.append(thresh2)
    
    # Method 3: Otsu's thresholding
    _, thresh3 = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_methods.append(thresh3)
    
    # Method 4: Simple thresholding with calculated threshold
    mean_val = np.mean(resized)
    _, thresh4 = cv2.threshold(resized, mean_val, 255, cv2.THRESH_BINARY)
    thresh_methods.append(thresh4)
    
    return thresh_methods, resized

def extract_text_multiple_configs(image):
    """Try multiple OCR configurations and return the best result"""
    configs = [
        # Config 1: Standard alphanumeric with spaces allowed
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
        
        # Config 2: Single line, no spaces
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        
        # Config 3: Single word
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        
        # Config 4: Treat as single character block
        r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        
        # Config 5: Default with less restrictions
        r'--oem 3 --psm 6'
    ]
    
    results = []
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config).strip()
            results.append(text)
        except:
            results.append("")
    
    return results

def clean_and_validate_text(text_list):
    """Clean and validate extracted text"""
    cleaned_results = []
    
    for text in text_list:
        if not text:
            continue
            
        # Remove unwanted characters and clean
        cleaned = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove spaces for compact format
        no_spaces = re.sub(r'\s', '', cleaned)
        
        cleaned_results.extend([cleaned, no_spaces])
    
    return cleaned_results

def detect_plate_patterns(text_list):
    """Detect various license plate patterns"""
    patterns = [
        # UK format: 2 letters + 2 digits + 3 letters
        r'[A-Z]{2}\d{2}[A-Z]{3}',
        
        # US format variations
        r'[A-Z]{3}\d{4}',  # 3 letters + 4 digits
        r'\d{3}[A-Z]{3}',  # 3 digits + 3 letters
        r'[A-Z]{2}\d{5}',  # 2 letters + 5 digits
        r'\d{2}[A-Z]{4}',  # 2 digits + 4 letters
        
        # European formats
        r'[A-Z]{1,3}\d{1,4}[A-Z]{1,3}',  # Mixed letters and numbers
        
        # General patterns
        r'[A-Z0-9]{5,8}',  # 5-8 alphanumeric characters
        r'\d{4,6}',        # 4-6 digits only
        r'[A-Z]{4,6}'      # 4-6 letters only
    ]
    
    matches = []
    for text in text_list:
        for pattern in patterns:
            found = re.findall(pattern, text)
            matches.extend(found)
    
    return matches

def score_plate_candidate(candidate):
    """Score plate candidates based on typical characteristics"""
    if not candidate or len(candidate) < 4:
        return 0
    
    score = 0
    
    # Length scoring (most plates are 5-8 characters)
    if 5 <= len(candidate) <= 8:
        score += 10
    elif 4 <= len(candidate) <= 9:
        score += 5
    
    # Mixed alphanumeric is good
    has_letters = bool(re.search(r'[A-Z]', candidate))
    has_numbers = bool(re.search(r'\d', candidate))
    
    if has_letters and has_numbers:
        score += 15
    elif has_letters or has_numbers:
        score += 5
    
    # Penalize too many repeating characters
    if len(set(candidate)) < len(candidate) * 0.5:
        score -= 5
    
    # Bonus for common patterns
    if re.match(r'[A-Z]{2}\d{2}[A-Z]{3}', candidate):  # UK format
        score += 20
    elif re.match(r'[A-Z]{3}\d{4}', candidate):  # Common US format
        score += 15
    
    return score

# Load model
model = YOLO(r"E:\car_plate_detect\models\best.pt")

# Load image
img_path = r"E:\car_plate_detect\Dataset\why-are-number-plates-yellow-and-white.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Error: Could not load image. Check the file path.")
    exit()

print(f"Processing image: {img_path}")

# Run inference
results = model(img)

# Process results
plate_count = 0
for r in results:
    boxes = r.boxes
    if boxes is None:
        continue
        
    for box in boxes:
        plate_count += 1
        print(f"\n--- Processing Plate {plate_count} ---")
        
        # Get bounding box coordinates
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        
        # Add some padding to the crop
        padding = 5
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(img.shape[1], xmax + padding)
        ymax = min(img.shape[0], ymax + padding)
        
        # Crop plate region
        plate_crop = img[ymin:ymax, xmin:xmax]
        
        if plate_crop.size == 0:
            print("Warning: Empty plate crop, skipping...")
            continue
        
        # Preprocess for OCR
        thresh_images, resized = preprocess_plate_image(plate_crop)
        
        all_candidates = []
        
        # Try OCR on all preprocessed images
        for i, thresh in enumerate(thresh_images):
            print(f"Trying preprocessing method {i+1}...")
            
            # Extract text with multiple configurations
            text_results = extract_text_multiple_configs(thresh)
            
            # Clean and validate text
            cleaned_texts = clean_and_validate_text(text_results)
            
            # Detect patterns
            patterns = detect_plate_patterns(cleaned_texts)
            
            # Add all results to candidates
            all_candidates.extend(cleaned_texts + patterns)
        
        # Remove duplicates and empty strings
        all_candidates = list(set([c for c in all_candidates if c and len(c) >= 3]))
        
        # Score and rank candidates
        scored_candidates = [(candidate, score_plate_candidate(candidate)) 
                           for candidate in all_candidates]
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"All candidates with scores:")
        for candidate, score in scored_candidates[:10]:  # Show top 10
            print(f"  '{candidate}' (score: {score})")
        
        # Select best candidate
        if scored_candidates:
            best_plate = scored_candidates[0][0]
            best_score = scored_candidates[0][1]
            print(f"\n DETECTED PLATE: '{best_plate}' (confidence score: {best_score})")
        else:
            print("\n No valid plate number detected")
        
        # Optional: Show images for debugging
        show_debug = True  # Set to False to disable image display
        if show_debug:
            # Show original crop
            cv2.imshow(f'Plate {plate_count} - Original', plate_crop)
            
            # Show best preprocessing result
            if thresh_images:
                cv2.imshow(f'Plate {plate_count} - Processed', thresh_images[0])
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

print(f"\nProcessing complete. Found {plate_count} plates.")