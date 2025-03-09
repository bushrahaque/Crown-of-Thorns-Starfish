import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ultralytics import YOLO
import pandas as pd
import os
import yaml

PERFORMANCE_OUTPUT_PATH = "/Users/bushra/Documents/STA2453/cots/model/outputs/ensemble_performance.txt"
YOLO_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/yolov8_tuned.pt"
YOLO_DATASET_YAML = "/Users/bushra/Documents/STA2453/cots/model/yolo_data_config.yaml"

def evaluate_classifier(model, data_loader, device):
    """Evaluates the binary classifier performance on test data."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute binary classification metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="binary"),
        "Recall": recall_score(y_true, y_pred, average="binary"),
        "F1 Score": f1_score(y_true, y_pred, average="binary")
    }

    return metrics

def evaluate_yolo():
    """Computes mAP for YOLO, using all test images."""
    model = YOLO(YOLO_MODEL_PATH)

    # Load YAML to get the file paths
    with open(YOLO_DATASET_YAML, 'r') as yaml_file:
        dataset_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # test_images = dataset_yaml['train'] + dataset_yaml['val']  # Use both train and val images for evaluation
    # test_images = dataset_yaml['val']

    # # Prepare YOLO test set for evaluation
    # test_data = [(str(img_data['image']), str(img_data['label'])) for img_data in test_images if os.path.exists(str(img_data['label']))]
    # print(test_data[0])

    # if not test_data:
    #     print("No valid labeled test images found!")
    #     return {"mAP@0.5": 0, "mAP@0.5:0.95": 0}

    # Perform YOLO evaluation
    results = model.val(data=YOLO_DATASET_YAML, split="val", conf=0.5)

    return {
        "mAP@0.5": results.box.map50,  # mAP@0.5
        "mAP@0.5:0.95": results.box.map  # mAP@0.5:0.95
    }


def save_performance(binary_metrics, yolo_metrics):
    """Saves all performance metrics to a text file."""
    with open(PERFORMANCE_OUTPUT_PATH, "w") as f:
        for key, value in {**binary_metrics, **yolo_metrics}.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"Performance metrics saved in {PERFORMANCE_OUTPUT_PATH}")


if __name__ == "__main__":
    yolo_metrics = evaluate_yolo()

    save_performance({}, yolo_metrics)

