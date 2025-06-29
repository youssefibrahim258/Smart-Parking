import pickle
from skimage.transform import resize
import numpy as np
import cv2

Empty = True
NOT_EMPTY = False

model = pickle.load(open(r"Y:\Graduation project\ParkDetect\ParkingDetector\Model\SVM_model", 'rb'))

def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = resize(spot_bgr , (15,15,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = model.predict(flat_data)

    if y_output == 0:
        return Empty
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):

    (totalLabels , label_ids , values , centroid) = connected_components

    slots = []

    coef = 1

    for i in range(1 , totalLabels):

        # Extract the coordinates...
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i , cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1,y1,w,h])

    return slots