# Author: Finn Purcell
# Last Modified: 27/08/2026

import os
import cv2
from matplotlib import pyplot as plt
from skimage.filters import threshold_niblack, threshold_sauvola, threshold_otsu
from skimage.util import img_as_float, img_as_ubyte #for conversion

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


def binarise_image(image, threshold=threshold_otsu, windowSize = 15, k=0.2):
    """converts to binary image"""
    image = img_as_float(image) #convert to use skilearn thresholding
    th = threshold(image)#, windowSize, k)
    binary_im = image > th

    cv_compat_binary_im = img_as_ubyte(binary_im)
    show_image(cv_compat_binary_im, title = "binary_image", cmap='binary')

    return img_as_ubyte(binary_im) #convert back to openCV

def connectedComponentAnalysis(image):
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 4)

    # Make a copy to draw on
    marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Start at 1 because label 0 is the background
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Draw bounding box
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

    return num_labels, labels, stats, centroids

def run_task2(image_path, config):
    # TODO: Implement task 2 here
    output_path = f"output/task2/result.txt"
    save_output(output_path, "Task 2 output", output_type='txt')

if __name__ == "__main__":
    image = read_im_gs("images/lcd2.png")
    binary = binarise_image(image, threshold_sauvola)
    connectedComponentAnalysis(binary)

#save_output("assignment/lastname_firstname_12345678/output/task2/task2output.txt", "bruh")