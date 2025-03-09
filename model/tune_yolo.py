import os
import pandas as pd
from ultralytics import YOLO

# Paths
YOLO_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/yolov8n.pt"  # Start with small YOLO model
TRAIN_CSV = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train_split.csv"
YOLO_DATASET_YAML = "/Users/bushra/Documents/STA2453/cots/model/yolo_data_config.yaml"
TUNED_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/yolov8_tuned.pt"


def tune_yolo():
    """Tunes YOLO on the training set."""

    print("Starting YOLO Hyperparameter Tuning...")
    model = YOLO(YOLO_MODEL_PATH)

    model.tune(data=YOLO_DATASET_YAML, epochs=50, iterations=100, device="mps")  # Run tuning

    model.save(TUNED_MODEL_PATH)
    print(f"YOLO tuning complete! Tuned model saved at {TUNED_MODEL_PATH}")

if __name__ == "__main__":
    tune_yolo()
