# Author: Finn Purcell
# Last Modified: 5/10/2026
# Contains training code for YOLO models

from ultralytics import YOLO
import csv

try:
    import wandb
except ImportError:
    pass


def train_model(DATA_YAML = "thermo_yolo/data.yaml", EPOCHS = 100, BATCH = 32, IMGSZ = 600, BMODEL="yolov8s-pose.pt"):
    """trains a single YOLO model and returns performance on test set"""
    yolo = YOLO(BMODEL)

    yolo.train(
        data = DATA_YAML,
        epochs = EPOCHS,
        batch = BATCH,
        imgsz = IMGSZ,
        patience = 15,
        device='cuda',
    )

    results = yolo.val(data=DATA_YAML, split="test")

    return results


def train_many(csv_file = "thermo_models.csv"):
    """completes a grid search of models trained using different YOLO configurations based on test set and saves results to csv."""
    batch_sizes = [8, 16, 32]
    im_sizes = [512, 640, 700]
    models = ["yolov8n.pt"]
    alreadyTrained = 0 #used to continue interupted sequences 
    trained_count = 0  #^^^     ^^^     ^^^     ^^^     ^^^

    for batchSize in batch_sizes:
        for imSize in im_sizes:
            for model in models:
                print(f"\nTraining: model={model}, batch={batchSize}, imgsz={imSize}")

                if trained_count < alreadyTrained:
                    print("Already trained model")
                else:
                    results = train_model(EPOCHS=800, BATCH=batchSize, IMGSZ=imSize, BMODEL=model)

                     # Extract validation metrics
                    box = results.box
                  

                    row = [
                        model,
                        batchSize,
                        imSize,
                        200,

                        box.mp,
                        box.mr,
                        box.map50,
                        box.map
                    ]

                    # Append results immediately
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                    print("Results saved to CSV.")
                trained_count += 1

if __name__ == "__main__":
    train_many()