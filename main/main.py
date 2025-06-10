import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


mask_path = r"Y:\Graduation project\ParkDetect\main\mask_1920_1080.png"
video_path = r"Y:\Graduation project\ParkDetect\Data\parking_1920_1080.mp4"


mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

width = mask.shape[1]
region_width = width // 4

regions = {"A": [], "B": [], "C": [], "D": []}
for spot in spots:
    x, y, w, h = spot
    if x < region_width:
        regions["A"].append(spot)
    elif x < 2 * region_width:
        regions["B"].append(spot)
    elif x < 3 * region_width:
        regions["C"].append(spot)
    else:
        regions["D"].append(spot)

spots_status = {region: [None] * len(spots) for region, spots in regions.items()}
diffs = {region: [None] * len(spots) for region, spots in regions.items()}

previous_frame = None
frame_nmr = 0
step = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for region, spots in regions.items():
            for i, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[region][i] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if frame_nmr % step == 0:
        for region, spots in regions.items():
            arr_ = range(len(spots)) if previous_frame is None else [j for j in np.argsort(diffs[region]) if
                                                                     diffs[region][j] / np.amax(diffs[region]) > 0.4]
            for i in arr_:
                x1, y1, w, h = spots[i]
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spots_status[region][i] = empty_or_not(spot_crop)

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    available_spots = {}
    for region, spots in regions.items():
        free_spots = sum(spots_status[region])
        available_spots[region] = (free_spots, len(spots))

        for i, (x1, y1, w, h) in enumerate(spots):
            color = (0, 255, 0) if spots_status[region][i] else (0, 0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # Draw region boundaries
    for i, region in enumerate(["A", "B", "C", "D"]):
        x_start = i * region_width
        x_end = (i + 1) * region_width if i < 3 else width
        frame = cv2.rectangle(frame, (x_start, 0), (x_end, mask.shape[0]), (255, 255, 0), 2)
        cv2.putText(frame, region, (x_start + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.rectangle(frame, (50, 20), (600, 150), (0, 0, 0), -1)
    y_offset = 40
    for region, (free, total) in available_spots.items():
        text = f"Region {region}: {free} / {total} available"
        cv2.putText(frame, text, (60, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30

    frame = cv2.resize(frame, (960, 540))
    cv2.imshow('Parking Status', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
