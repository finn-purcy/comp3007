# Author: Finn Purcell
# Last Modified: 29/08/2026
# contains task2 code for sgmenting lcd displays/thermometer readings.

import os
import cv2
from matplotlib import pyplot as plt
from skimage.filters import threshold_niblack, threshold_sauvola, threshold_otsu
from skimage.util import img_as_float, img_as_ubyte #for conversion
import numpy as np
import sys
from utilities import *

def read_im_gs(image_path):
    """reads in an image from specified path and converts to greyscale"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def preprocess_image(image, threshold=threshold_otsu, windowSize = 15, k=0.2):
    """applies median blur and converts to binary image"""
    image = cv2.resize(image, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.medianBlur(image, 5)
    image = img_as_float(image) #convert to use skilearn thresholding

    if threshold == threshold_otsu:
        th = threshold(image)
    elif threshold == None:
        return img_as_ubyte(image)
    else:
        th = threshold(image, windowSize, k) #statistical thresholds require additional params

    binary_im = image > th

    cv_im = img_as_ubyte(binary_im) #convert back to openCV
    

    return cv_im 

def boxes_close(box1, box2, distance):
    """logic used to determine whether detected blobs are close enough to merge"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    right1 = x1 + w1
    bottom1 = y1 + h1

    right2 = x2 + w2
    bottom2 = y2 + h2

    horizontal_gap = max(x1 - right2, x2 - right1, 0)
    vertical_gap = max(y1 - bottom2, y2 - bottom1, 0)

    return horizontal_gap <= distance and vertical_gap <= distance

def merge_blobs(blobs, distance):
    """merges nearby blobs to account for distances between digit segments"""

    parent = list(range(len(blobs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    # find blobs that should be merged
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            if boxes_close(blobs[i], blobs[j], distance):
                union(i, j)

    # group blobs
    groups = {}

    for i in range(len(blobs)):
        root = find(i)
        groups.setdefault(root, []).append(blobs[i])

    # create merged bounding boxes
    merged = []

    for group in groups.values():

        x_min = min(x for x, y, w, h in group)
        y_min = min(y for x, y, w, h in group)

        x_max = max(x + w for x, y, w, h in group)
        y_max = max(y + h for x, y, w, h in group)

        merged.append([
            x_min,
            y_min,
            x_max - x_min,
            y_max - y_min
        ])

    return merged

def order_blobs(blobs):
    """orders blobs for correct file output"""
    sysBlobs = []
    diaBlobs = []
    pulseBlobs = []

    sysY = 1000 #top number
    pulseY = 0 #bottom number
    yTolerance = 150

    #first pass finds correct y values
    for blob in blobs:
        blobY = blob[1]

        if blobY < sysY:
            sysY = blobY
        if blobY > pulseY:
            pulseY = blobY

    #second pass groups each blob
    for blob in blobs:
        blobY = blob[1]

        if abs(blobY-sysY) < yTolerance:
            sysBlobs.append(blob)
        elif abs(blobY-pulseY) < yTolerance:
            pulseBlobs.append(blob)
        else:
            diaBlobs.append(blob)

    #within each group order blobs
    sysBlobs.sort(key=lambda x: x[0])
    diaBlobs.sort(key=lambda x: x[0])
    pulseBlobs.sort(key=lambda x: x[0])

    orderedBlobs = sysBlobs + diaBlobs + pulseBlobs

    return orderedBlobs

        

def connectedComponentAnalysis(image, minh = 50, maxh = 350, minw = 10, mina = 45, maxa = 100000):
    num_labels, ids, stats, centroids = cv2.connectedComponentsWithStats(image, 8)

    # copy to draw on
    marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    blobs = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if (area > mina) and (area < maxa):
            blobs.append([x, y, w, h])
            digitMask = (ids == i).astype("uint8") * 255

    # Merge nearby blobs
    merged_blobs = merge_blobs(merge_blobs(blobs, distance=20), distance=35) #done twice tends to perform better


    filtered_blobs = []

    #apply filtering to surrounding blobs
    for x, y, w, h in merged_blobs: 
        if h > minh and w > minw and x > 150 and x < 950 and y > 120:
            if h < maxh:
                filtered_blobs.append([x,y,w,h])
            elif w > 60: # Split larger blobs that have joined incorrectly
                filtered_blobs.append([x, y, w, int(h/2)])
                filtered_blobs.append([x, int(y+h/2), w, int(h/2)])
        else:
            cv2.rectangle(marked_image, (x, y), (x+w,y+h), (255,0, 0), 2)


    for x, y, w, h in filtered_blobs:
        # draw bounding box
        cv2.rectangle(
            marked_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Draw component number
        cv2.putText(
            marked_image,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return filtered_blobs

def extractDigits(blobs, originalImage, processedImage, originalFileName, output, t4mode = False):
    images = []

    originalY, originalX = originalImage.shape[:2]
    processedY, processedX = processedImage.shape[:2]

    #counteract effect of resize in preprocessing
    scaleX = originalX/processedX
    scaleY = originalY/processedY

    digitNumber = 0

    for blob in blobs:
        digitNumber += 1
        fileName = f"d{digitNumber}"

        x, y, w, h = blob

        padX = 25
        padY = 25

        if h < 130: #manual adjustment for incorrect reading of bottom digits
            bottom_boost = 50
        else:
            bottom_boost = 0

        scaledX1 = max(int((x-padX) * scaleX), 0)
        scaledX2 = min(int((x+w+padX) * scaleX), originalX)
        scaledY1 = max(int((y-padY) * scaleY), 0)
        scaledY2 = min(int((y+h+padY+bottom_boost) * scaleY), originalY)

        croppedImage=originalImage[scaledY1: scaledY2, scaledX1: scaledX2]
        images.append(croppedImage)

        if t4mode:
            save_output(f"task2/d{digitNumber}.png", croppedImage, output_type='image')
        else:
            save_output(f"{output}/{originalFileName.split('.')[0]}/d{digitNumber}.png", croppedImage, output_type='image')


def lcd_digit_extract(imagePath, fileName, output, t4mode = False):
    originalImage, gray = read_im_gs(imagePath)

    binary = preprocess_image(gray, threshold_sauvola, windowSize=15, k=0.025)
    digits = connectedComponentAnalysis(binary, maxa = 10000)
    digits = order_blobs(digits)

    extractDigits(digits, originalImage, binary, fileName, output, t4mode)

def thermo_find_level(imagePath, fileName, output, t4mode = False):
    originalImage, gray = read_im_gs(imagePath)

    hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    ppImage = cv2.resize(red_mask, (1000, 1000))
    edges = cv2.Canny(ppImage, 50, 150, apertureSize=5)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100,
        minLineLength=90, maxLineGap=20
    )

    ppImage = cv2.cvtColor(ppImage, cv2.COLOR_GRAY2BGR)

    currThermoY = 500
    thermoY = 500
    thermoX = 500

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            if abs(x1-x2) < 20 and x1 > 350 and x1 < 650:
     
                cv2.line(ppImage,(x1,y1),(x2,y2),(0,255,0),1)
                thermoY = np.min([y1, y2]) 

                if thermoY < currThermoY and thermoY > 200: #esnures that top most thermo reading kept
                    currThermoY = thermoY

                thermoX = x1


    originalY, originalX = originalImage.shape[:2]
    processedY, processedX = ppImage.shape[:2]

    #counteract effect of resize in preprocessing
    pad = 110
    scaleY = originalY/processedY

    thermEndLow = int((currThermoY-pad) * scaleY)
    thermEndHigh = int((currThermoY+pad) * scaleY)

    croppedImage = originalImage[max(thermEndLow, 0): min(thermEndHigh, originalY), 0: originalX]    

    if currThermoY < 500:
        if t4mode:
            save_output(f"task2/t.png", croppedImage, output_type='image')
        else:
            save_output(f"{output}/{fileName}/t.png", croppedImage, output_type='image')
    else:
        print("failed to detect a thermo line")

def run_task2(image_path, config, task4mode = False):
    image_path = os.path.abspath(image_path)
    imPaths = find_all_files(image_path)

    if task4mode: #adjust for directory shifting in task 4
        config = f"../../{config}"
    config_data = read_config(config, 2)
   
    for imPath, entry in imPaths:
        if "lcd" in imPath:
            lcd_digit_extract(imPath, entry, config_data['output_path'], task4mode)
        else:
            try:
                thermo_find_level(imPath, entry, config_data['output_path'], task4mode)
            except ValueError:
                print("failed to find thermometer level")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("IMPROPER USE CASE!")
    else:
        run_task2(sys.argv[1], sys.argv[2])