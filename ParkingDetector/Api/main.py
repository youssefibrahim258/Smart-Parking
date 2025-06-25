from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from util import get_parking_spots_bboxes, empty_or_not
from skimage.transform import resize
import io

app = FastAPI()

mask = cv2.imread("mask_1920_1080.png", 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
width = mask.shape[1]
region_width = width // 4

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

@app.post("/status")
async def get_status(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    output = {}

    for region, spots in regions.items():
        empty = 0
        for spot in spots:
            x, y, w, h = spot
            crop = frame[y:y+h, x:x+w]
            if empty_or_not(crop):
                empty += 1
        output[region] = {
            "empty": empty,
            "total": len(spots)
        }

    return output
