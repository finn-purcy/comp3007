# Author: Finn J. Purcell
# Last Modified: 20-09-2026

import os
from ultralytics import YOLO
import glob
import wandb
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#wandb.login(key="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")

class OriginalImage():
    def __init__(self, image):
        self.image = image
        self.x = image.shape[1]
        self.y = image.shape[0]

        self.dx = self.x/500
        self.dy = self.y/500

    def relocatePoints(self, newPoints):
        """takes resized points and tranlates to points on original image"""
        transformX = newPoints[0]
        transformY = newPoints[1]

        originalX = int(transformX*self.dx)
        originalY = int(transformY*self.dy)

        return (originalX, originalY)

    def perspectiveTransform(self, transformCorners):
        """creates transformed image"""
        
        originalCorners = np.float32([[0,self.y], [self.x,self.y], [self.x, 0], [0,0]])

        matrix = cv2.getPerspectiveTransform(transformCorners.cpu().numpy(), originalCorners)
        result  =cv2.warpPerspective(self.image, matrix, (self.x,self.y)) 

        save_output(f"output/task1/testingT1.png", result, output_type='image')


        

def preprocess_image(image):
    """applies same preprocessing steps as training images for best model performance
    image = opencv image"""
    resize = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)



def train_model():
    DATA_YAML = "data/data.yaml"

    EPOCHS = 100
    BATCH = 128
    IMGSZ = 500

    yolo = YOLO("yolov8s-obb.pt")

    yolo.train(
        data = DATA_YAML,
        epochs = EPOCHS,
        batch = BATCH,
        imgsz = IMGSZ,
        patience = 15,
        device='cuda',
    )

    return yolo

def predict(model, imagePath):
    imPaths = []


    for entry in os.listdir(imagePath):
        full_path = os.path.join(imagePath, entry)
        imPaths.append((full_path, entry))
            
    for imPath, entry in imPaths:
        out = model(imPath, device="CPU")
        Image.fromarray(out[0].plot()[:,:,::-1])

def inference(ptFile = "runs/obb/train2/weights/best.pt"):
    yolo = YOLO(ptFile)

    imRead = cv2.imread("data/test/images/IMG_7585_jpeg.rf.38d6ea90f4dfdcb578e3bd4c4dc4d4f0.jpg")
    Image = OriginalImage(imRead)

    results = yolo("data/test/images/IMG_7585_jpeg.rf.38d6ea90f4dfdcb578e3bd4c4dc4d4f0.jpg")
    print(f"results: {results}")
    for result in results:
        obb = result.obb
        for corners, confidence, cls in zip(
            obb.xyxyxyxy,
            obb.conf,
            obb.cls
        ):
            print("Corners:", corners)
            print("Confidence:", confidence)
            print("Class:", cls)
            Image.perspectiveTransform(corners)


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


def run_task1(image_path, config):
    # TODO: Implement task 1 here
    print("TRAINING")
    model = train_model()
    print("INFERENCE")

if __name__ == "__main__":
    #run_task1("btuh", "tg")
    inference()
    """lab232-b01"""