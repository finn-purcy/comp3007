

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Finn J. Purcell
# Last Modified: 20-09-2026

import os
from ultralytics import YOLO
import glob
import wandb
import cv2
from PIL import Image
import numpy as np

wandb.login(key="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")

class OriginalImage():
    def __init__(self, image):
        self.image = image
        self.x = image[1]
        self.y = image[0]

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
        """creates transformed image
        transform corner must be bl, tl, br, tr in (y,x)"""
        originalCorners = np.float32([[0,0],[self.y, 0], [0,self.x], [self.y,self.x]])

        matrix = cv2.getPerspectiveTransform(transformCorners, originalCorners)
        result  =cv2.warpPerspective(self.image, matrix, (600,600))        


        

def preprocess_image(image):
    """applies same preprocessing steps as training images for best model performance
    image = opencv image"""
    resize = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)



def train_model():
    DATA_YAML = "data/data.yaml"

    EPOCHS = 30
    BATCH = 64
    IMGSZ = 500


    yolo = YOLO("yolov8n-obb.pt")

    yolo.train(
        data = DATA_YAML,
        epochs = EPOCHS,
        batch = BATCH,
        imgsz = IMGSZ,
        patience = 10,
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
    run_task1("btuh", "tg")

    """lab232-b01"""