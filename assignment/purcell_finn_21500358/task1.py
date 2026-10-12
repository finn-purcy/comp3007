# Author: Finn J. Purcell
# Last Modified: 29/09/2026
# Code for required location/identification inference and gemoetrical transformation

from ultralytics import YOLO
import cv2
import numpy as np
import sys
from utilities import *



class OriginalImage():
    def __init__(self, image):
        self.image = image
        self.x = image.shape[1]
        self.y = image.shape[0]

        self.dx = self.x/900
        self.dy = self.y/900

        self.processed = self.preprocess()

    def preprocess(self):
        """imitates resize undertaken in Roboflow preprocessing"""
        return (cv2.resize(self.image, (900, 900), interpolation=cv2.INTER_LINEAR))

    def relocatePoints(self, newPoints):
        """takes resized points and tranlates to points on original image"""
        transformX = newPoints[0]
        transformY = newPoints[1]

        originalX = int(transformX*self.dx)
        originalY = int(transformY*self.dy)

        return (originalX, originalY)

    def perspectiveTransform(self, transformCorners):
        """creates transformed image"""
        for corners in transformCorners.cpu().numpy():
            relocateCorners = np.float32([self.relocatePoints(corner) for corner in corners])

        #calculate appropriate dimensions for segment image
        newX = int(np.sqrt((relocateCorners[0][0]-relocateCorners[1][0])**2+(relocateCorners[0][1]-relocateCorners[1][1])**2))
        newY = int(np.sqrt((relocateCorners[2][0]-relocateCorners[1][0])**2+(relocateCorners[2][1]-relocateCorners[1][1])**2))

        originalCorners = np.float32([[0,0], [newX,0], [newX, newY], [0,newY]])


        matrix = cv2.getPerspectiveTransform(relocateCorners, originalCorners)
        result  =cv2.warpPerspective(self.image, matrix, (newX,newY)) 

        if result.shape[1] > result.shape[0]: #if image is wider than it is long
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

        return result

def inference_and_transform(ptFile = "rotate_small.pt", imPath = "yolo_keypoints/test/images/20260815_103114_jpg.rf.8a9ce5921184effb529bff2cf2ef8bde.jpg", output = "output/task1", count = [0,0]):
    """completes task 1 tasks for a single specified image"""
    yolo = YOLO(ptFile)

    imRead = cv2.imread(imPath)
    Image = OriginalImage(imRead)

    results = yolo(Image.processed)
    for result in results:
        keypoints = result.keypoints.xy
        cls = result.boxes.cls
        try:
            if int(cls) == 0:
                fileName = "lcd" + str(count[0])
                count[0] +=1
            else:
                fileName = "thermo" + str(count[1])
                count[1] +=1

            outIm = Image.perspectiveTransform(keypoints)

            save_output(f"{output}/{fileName}.png", outIm, output_type='image')
        
        
        except ValueError: #no detection
            count = count

    return count

def run_task1(image_path, config, task4mode = False):
    #returns all images in given directory
    imPaths = find_all_files(image_path)

    count = [0, 0] #keeps count of num of lcd and thermo images

    config_data = read_config(config, 1)

    if task4mode:
        inference_and_transform(imPath=imPaths[0][0], count = count, output=f"{image_path}/task1")

    else:
        if len(imPaths) != 0:
            for imPath, entry in imPaths:
                count = inference_and_transform(ptFile=config_data['model_path'] ,imPath=imPath, count = count, output=config_data['output_path'])
        else:
            print("No Images Detected in Specified Directory")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("IMPROPER USE CASE!")
    else:
        run_task1(sys.argv[1], sys.argv[2])

   