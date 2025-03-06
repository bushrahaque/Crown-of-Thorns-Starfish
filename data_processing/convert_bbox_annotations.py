import os
import pandas as pd
import cv2

class BBoxAnnotationConverter:
    def __init__(self, csv_path, images_dir, labels_dir):
        """
        Initializes the converter with paths.
        
        Input:
        csv_path (str): Path to the CSV file containing annotations.
        images_dir (str): Directory where image files are stored.
        labels_dir (str): Directory where formatted annotation text files will be saved.
        """
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        os.makedirs(self.labels_dir, exist_ok=True)

    def _normalize_bbox(self, bbox, img_width, img_height):
        """
        Normalizes bounding box coordinates.

        Input:
        bbox (dict): Bounding box info with 'x', 'y', 'width', 'height'.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

        Returns:
        str: Formatted annotation.
        """
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w /= img_width
        h /= img_height

        return f'0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}'

    def convert_annotations(self):
        """
        Converts the CSV annotations into the described format and saves them as text files.
        """
        df = pd.read_csv(self.csv_path)

        for _, row in df.iterrows():
            image_id = row['video_frame']
            img_path = os.path.join(self.images_dir, f'{image_id}.jpg')
            txt_filename = os.path.join(self.labels_dir, f'{image_id}.txt')

            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                print(f'Warning: Image {img_path} not found, skipping.')
                continue

            # Load image to get dimensions; skip if any error
            img = cv2.imread(img_path)
            if img is None:
                print(f'Error: Could not load {img_path}, skipping.')
                continue

            img_height, img_width = img.shape[:2]

            # Check if annotations exist; skip if no COTS detected
            if pd.isna(row['annotations']) or row['annotations'] == '[]':
                continue

            # Parse bounding box data
            annotations = eval(row['annotations'])

            with open(txt_filename, 'w') as f:
                for ann in annotations:
                    f.write(self._normalize_bbox(ann, img_width, img_height) + '\n')

        print('Success! The formatted annotation text files have been saved in:', self.labels_dir)

