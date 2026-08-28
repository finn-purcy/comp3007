# Author: Finn Purcell
# Last Modified: 27/08/2026

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
    return gray

def show_image(image, title='test', cmap = 'gray'):
    plt.figure()
    plt.imshow(image, cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()
    plt.savefig(f'test_out/{title}.png')


def preprocess_image(image, threshold=threshold_otsu, windowSize = 15, k=0.2):
    """applies median blur and converts to binary image"""
    image = cv2.resize(image, (1000, 1000))
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.medianBlur(image, 5)
    image = img_as_float(image) #convert to use skilearn thresholding
    if threshold == threshold_otsu:
        th = threshold(image)
    else:
        th = threshold(image, windowSize, k) #statistical thresholds require additional params
    binary_im = image > th

    cv_im = img_as_ubyte(binary_im) #convert back to openCV

    #kernel = np.ones((5, 5), np.uint8)
    #combined = cv2.dilate(cv_im, kernel, iterations=1)

    #cv_im = cv2.medianBlur(cv_im, 7)
    
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

    return cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)

def run_task2(image_path, config):
    # TODO: Implement task 2 here
    output_path = f"output/task2/result.txt"
    save_output(output_path, "Task 2 output", output_type='txt')

if __name__ == "__main__":
    image = read_im_gs("images/lcd2.png")

    binary = preprocess_image(image, threshold_sauvola, windowSize=15, k=0.025)
    markedImage = connectedComponentAnalysis(binary)
    
    print(np.unique(binary))

#save_output("assignment/lastname_firstname_12345678/output/task2/task2output.txt", "bruh")