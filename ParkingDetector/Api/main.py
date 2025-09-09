from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
from util import get_parking_spots_bboxes, empty_or_not
from skimage.transform import resize
import io

app = FastAPI()

# Global variable to store the latest result
last_status = None

# Load and process the parking mask
mask = cv2.imread("mask_1920_1080.png", 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
width = mask.shape[1]
region_width = width // 4

# Divide spots into regions a-d based on horizontal position
regions = {"a": [], "b": [], "c": [], "d": []}
for spot in spots:
    x, y, w, h = spot
    if x < region_width:
        regions["a"].append(spot)
    elif x < 2 * region_width:
        regions["b"].append(spot)
    elif x < 3 * region_width:
        regions["c"].append(spot)
    else:
        regions["d"].append(spot)

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "Parking spot detection API is running."}

# Main endpoint to check parking status
@app.post("/status")
async def get_status(file: UploadFile = File(...)):
    global last_status

    contents = await file.read()

    # Convert file to image
    try:
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    output = {}

    # Process each region
    for region, spots_list in regions.items():
        empty = 0
        for spot in spots_list:
            x, y, w, h = spot
            crop = frame[y:y+h, x:x+w]
            if empty_or_not(crop):
                empty += 1
        output[region] = {
            "empty": empty,
            "total": len(spots_list)
        }

    # Save latest result
    last_status = output

    return output

# New endpoint to return the last parking status
@app.get("/last-status")
def get_last_status():
    if last_status is None:
        raise HTTPException(status_code=404, detail="No status available yet.")
    return last_status
