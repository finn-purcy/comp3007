# Author: Finn Purcell
# Last Modified: 9/10/2026
# Digit classification (0-9) using KNN on binarised images

import os
import sys
import cv2
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from utilities import *
from ultralytics import YOLO


# Binarised images are shrunk to this size, so each image becomes a
# 32*32 = 1024 long vector of 0s and 1s.
IMG_SIZE = 32
MODEL_PATH = "best_knn.joblib"


def binarise_image(image):
    """
    BGR image -> flat vector of 0/1 of length IMG_SIZE * IMG_SIZE.
    Also returns the 0/255 binary image so it can be saved for checking.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Shrink first (INTER_AREA is the right choice for downsizing), then
    # blur slightly to suppress noise before thresholding.
    grey = cv2.resize(grey, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)

    # Otsu picks the threshold per image, so it copes with lighting changes.
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Make polarity consistent: the digit (foreground) is normally the
    # minority of pixels, so if most pixels are white, invert.
    #if np.mean(binary == 255) > 0.5:
    #    binary = 255 - binary

    vector = (binary.flatten() // 255).astype(np.uint8)
    return vector, binary


def load_data_from_directory(dirpath):

    vectors = []
    labels = []

    for i in range(10):
        class_dir = os.path.join(dirpath, str(i))

        if not os.path.isdir(class_dir):
            print(f"WARNING: Directory not found: {class_dir}")
            continue

        print(f"Loading class {i}")

        for entry in os.listdir(class_dir):
            full_path = os.path.join(class_dir, entry)

            # Skip directories
            if not os.path.isfile(full_path):
                continue

            image = cv2.imread(full_path)

            if image is None:
                print(f"WARNING: Could not read image: {full_path}")
                continue

            vector, _ = binarise_image(image)

            vectors.append(vector)
            labels.append(i)

    if len(vectors) == 0:
        raise ValueError(
            f"No readable images found in directory: {dirpath}"
        )

    print(f"Loaded {len(vectors)} images from {dirpath}")

    return np.array(vectors, dtype=np.uint8), np.array(labels, dtype=np.int64)


def make_knn(k):
    # Hamming distance = fraction of pixels that differ, which is the natural
    # distance between binary vectors.
    return KNeighborsClassifier(
        n_neighbors=k, metric="hamming", algorithm="brute", weights="distance"
    )


def train_KNN(k_values=(1, 3, 5, 7, 9, 11), n_splits=5):

    # Load datasets
    X_train, y_train = load_data_from_directory("Task 3 Train Data/train")
    X_valid, y_valid = load_data_from_directory("Task 3 Train Data/valid")
    X_test, y_test = load_data_from_directory("Task 3 Train Data/test")

    # Combine training and validation data
    X_all = np.concatenate([X_train, X_valid])
    y_all = np.concatenate([y_train, y_valid])

    # Stratified folds preserve class proportions
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)

    # Cross-validation to choose k
    mean_accs = {}

    for k in k_values:
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
            knn = make_knn(k)
            knn.fit(X_all[train_idx], y_all[train_idx])
            fold_accuracies.append(knn.score(X_all[val_idx], y_all[val_idx]))

        mean_accs[k] = float(np.mean(fold_accuracies))
        std_acc = float(np.std(fold_accuracies, ddof=1))
        print(f"k={k}: CV accuracy {mean_accs[k]:.4f} +/- {std_acc:.4f}")

    best_k = max(mean_accs, key=mean_accs.get)
    print(f"\nBest k: {best_k} (CV accuracy {mean_accs[best_k]:.4f})")

    # Fit final model on all train + validation data
    final_model = make_knn(best_k)
    final_model.fit(X_all, y_all)

    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

    # Evaluate the untouched test set only once
    test_acc = final_model.score(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    return test_acc


def inference_image(image_path, model_path):
    knn = joblib.load(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print("BAD IMAGE PATH DURING TASK 3 INFERENCE!")
        return None

    vector, binary = binarise_image(image)

    return int(knn.predict(vector.reshape(1, -1))[0])

def thermo_inference(image_path, ptFile):
    yolo = YOLO(ptFile)

    image = cv2.imread(image_path)
    resize = cv2.resize(image, (640, 640), cv2.INTER_LINEAR)

    results = yolo(resize)

    print(results)
    
    pass
            

def run_task3(image_path, config, task4mode = False):
    #train_KNN()
    imPaths = find_all_files(image_path)

    if task4mode:
        config = f"../../{config}"
    config_data = read_config(config, 3)

    if len(imPaths) != 0:
        for imPath in imPaths:

            if task4mode:
                prediction = {inference_image(imPath[0], f"../../{config_data['model_path']}")}
                save_output(f"task3/{imPath[0][imPath[0].find('task2/')+6:imPath[0].find('.png')]}.txt", str(prediction)[1:2])
            else:
                prediction = {inference_image(imPath[0], config_data['model_path'])}
                save_output(f"{config_data['output_path']}/{imPath[0][imPath[0].find('lcd'):imPath[0].find('.png')]}.txt", str(prediction)[1:2])
            #print(f"imPath[0]: {imPath[0]}\n.imPath[1]: {imPath[1]}")
    else:
        print("No Images Detected in Specified Directory")


if __name__ == "__main__":
    run_task3(sys.argv[1], sys.argv[2])