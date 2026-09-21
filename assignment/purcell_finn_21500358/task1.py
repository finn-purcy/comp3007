

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

wandb.login(key="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")


def train_model():
    DATA_YAML = "data/data.yaml"

    EPOCHS = 30
    BATCH = 32
    IMGSZ = 500


    yolo = YOLO("yolov8s-obb.pt")

    yolo.train(
        data = DATA_YAML,
        epochs = EPOCHS,
        batch = BATCH,
        imgsz = IMGSZ,
        patience = 5,
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