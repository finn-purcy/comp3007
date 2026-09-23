# Author: Finn Purcell
# Last Modified: 29/08/2026

import os
import cv2
from matplotlib import pyplot as plt
from skimage.filters import threshold_niblack, threshold_sauvola, threshold_otsu
from skimage.util import img_as_float, img_as_ubyte #for conversion
import numpy as np

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

def read_im_gs(image_path):
    """reads in an image from specified path and converts to greyscale"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def show_image(image, title='test', cmap = 'gray'):
    plt.figure()
    plt.imshow(image, cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.savefig(f'test_out/{title}.png')
    plt.show()
    plt.close()


def preprocess_image(image, threshold=threshold_otsu, windowSize = 15, k=0.2):
    """applies median blur and converts to binary image"""
    image = cv2.resize(image, (1000, 1000))
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
    
    show_image(cv_im, title = "binary_image", cmap='binary')

    return cv_im 

def boxes_close(box1, box2, distance):
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

        

def connectedComponentAnalysis(image):
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

        if (area > 50) and (area < 50000):
            blobs.append([x, y, w, h])
            digitMask = (ids == i).astype("uint8") * 255

    # Merge nearby blobs
    merged_blobs = merge_blobs(blobs, distance=30)

    filtered_blobs = []

    for x, y, w, h in merged_blobs:
        if h > 95:
            filtered_blobs.append([x,y,w,h])

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

    # Display
    show_image(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB), "blob_detection", cmap=None)

    return filtered_blobs

def extractDigits(blobs, originalImage, processedImage, originalFileName):
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

        x = int(x*scaleX)
        w = int(w*scaleX)
        y = int(y*scaleY)
        h = int(h*scaleY)

        pad = 20
        croppedImage=originalImage[max(y-pad, 0): min(y+h+pad, originalY), max(x-pad, 0): min(x+w+pad, originalX)]
        images.append(croppedImage)

        save_output(f"output/task2/{originalFileName}/d{digitNumber}.png", croppedImage, output_type='image')

        show_image(croppedImage, "digit", cmap=None)

def lcd_digit_extract(imagePath, fileName):
    originalImage, gray = read_im_gs(imagePath)

    binary = preprocess_image(gray, threshold_sauvola, windowSize=15, k=0.025)
    digits = connectedComponentAnalysis(binary)
    digits = order_blobs(digits)

    extractDigits(digits, originalImage, binary, fileName)

def thermo_find_level(imagePath, fileName):
    originalImage, gray = read_im_gs(imagePath)

    ppImage = preprocess_image(gray)
    edges = cv2.Canny(ppImage, 50, 150, apertureSize= 5)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=90,maxLineGap=20)

    ppImage = cv2.cvtColor(ppImage, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line
            if abs(x1-x2) < 10:
                cv2.line(ppImage,(x1,y1),(x2,y2),(0,255,0),2)
                thermoY = np.min([y1, y2])
                thermoX = x1
    
    originalY, originalX = originalImage.shape[:2]
    processedY, processedX = ppImage.shape[:2]

    #counteract effect of resize in preprocessing
    pad = 300
    scaleX = originalX/processedX
    scaleY = originalY/processedY

    thermEndY = int(thermoY * scaleY)
    thermEndX = int(thermoX * scaleX)

    croppedImage = originalImage[max(thermEndY-pad, 0): min(thermEndY+pad, originalY), max(thermEndX-pad, 0): min(thermEndX+pad, originalX)]    
    

    save_output(f"output/task2/{fileName}/t.png", croppedImage, output_type='image')

def run_task2(image_path, config):
    imPaths = []
    for entry in os.listdir(image_path):
        full_path = os.path.join(image_path, entry)
        imPaths.append((full_path, entry))
            
    for imPath, entry in imPaths:
        if "lcd" in imPath:
            print("A")
            lcd_digit_extract(imPath, entry)

        else:
            thermo_find_level(imPath, entry)


if __name__ == "__main__":
    run_task2("/home/21500358/comp3007/assignment/purcell_finn_21500358/output/task1", "config.txt")
