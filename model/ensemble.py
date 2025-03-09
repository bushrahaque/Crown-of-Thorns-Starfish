import torch
from ultralytics import YOLO
from binary_classifier import COTSClassifier, COTSDataset
import pandas as pd
import cv2
import torchvision.transforms as transforms
import yaml

# Paths
BINARY_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/binary_classifier.pth"
YOLO_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/yolov8_tuned.pt"
YOLO_DATASET_YAML = "/Users/bushra/Documents/STA2453/cots/model/yolo_data_config.yaml"
OUTPUT_TXT_PATH = "/Users/bushra/Documents/STA2453/cots/model/outputs/ensemble_predictions.txt"

# Load Binary Classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
binary_model = COTSClassifier().to(device)
binary_model.load_state_dict(torch.load(BINARY_MODEL_PATH))
binary_model.eval()

# Load YOLO Model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Data Transform for Binary Classifier
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load YAML to get the file paths
with open(YOLO_DATASET_YAML, "r") as yaml_file:
    dataset_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Extract train and validation image-label pairs
train_images = dataset_yaml["train"]
val_images = dataset_yaml["val"]

# Combine them for testing as you"re running inference on the full dataset
test_images = train_images + val_images

predictions = []

# Process each image in the test set
for img_data in test_images:
    img_path = img_data["image"]
    label_path = img_data["label"]

    # Load image for binary classification
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Binary classification
    with torch.no_grad():
        output = binary_model(image)
        _, pred = torch.max(output, 1)

    if pred.item() == 1:  # If COTS detected, run YOLO
        results = yolo_model(img_path)
        boxes = results[0].boxes
        pred_str = f"Image: {img_path}, Detections: {len(boxes)}"
    else:
        pred_str = f"Image: {img_path}, Detections: 0 (No Detection)"

    predictions.append(pred_str)
    print(pred_str)

# Save results to a text file
with open(OUTPUT_TXT_PATH, "w") as f:
    for line in predictions:
        f.write(line + "\n")

print(f"Ensemble predictions saved to {OUTPUT_TXT_PATH}")

