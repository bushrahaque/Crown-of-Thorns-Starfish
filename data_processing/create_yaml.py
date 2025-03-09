import pandas as pd
import yaml
import os

# File paths to your CSV files
train_csv = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train_split.csv"
test_csv = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/test_split.csv"

# Output path for the generated YAML
yaml_output_path = "/Users/bushra/Documents/STA2453/cots/model/yolo_data_config.yaml"

# Read CSVs
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Extract the image paths and labels
train_images = train_df["image_path"].tolist()
train_labels = train_df["label_path"].tolist()

test_images = test_df["image_path"].tolist()
test_labels = test_df["label_path"].tolist()

# Create YAML structure
dataset_yaml = {
    "train": [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)],
    "val": [{"image": img, "label": lbl} for img, lbl in zip(test_images, test_labels)],
    "nc": 1,         
    "names": ["COTS"]
}

# Save the YAML file
with open(yaml_output_path, "w") as yaml_file:
    yaml.dump(dataset_yaml, yaml_file, default_flow_style=False)

print(f"YAML file created at {yaml_output_path}")