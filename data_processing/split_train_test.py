import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
TRAIN_CSV_PATH = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train.csv"
OUTPUT_DIR = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/"

# Image and label directories
IMAGE_DIRS = [f"/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train_images/video_{i}" for i in range(3)]
LABEL_DIRS = [f"/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train_images/labels_{i}" for i in range(3)]

def get_file_path(image_id, video_id):
    """
    Returns the full image path and label path for a given image_id and video_id.
    """
    image_path = os.path.join(IMAGE_DIRS[video_id], f"{image_id}.jpg")
    label_path = os.path.join(LABEL_DIRS[video_id], f"{image_id}.txt")
    
    return image_path, label_path

def split_dataset(test_size=0.2, random_state=2453):
    """
    Splits train.csv into train/test splits and saves them as CSVs.
    """
    # Load full dataset
    df = pd.read_csv(TRAIN_CSV_PATH)

    # Map image_id to full image and label paths
    df["image_path"], df["label_path"] = zip(*df.apply(lambda row: get_file_path(row["video_frame"], row["video_id"]), axis=1))

    # Train-test split and save
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), index=False)

    print("Train-test split completed.")
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

if __name__ == "__main__":
    split_dataset()

