import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm
from performance_metrics import evaluate_classifier

# Paths
TRAIN_CSV = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train_split.csv"
TEST_CSV = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/test_split.csv"
OUTPUT_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/binary_classifier.pth"
PERFORMANCE_OUTPUT_PATH = "/Users/bushra/Documents/STA2453/cots/model/outputs/binary_classifier_performance.txt"

# Define Dataset
class COTSDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        label = 1 if self.data.iloc[idx]["annotations"] != "[]" else 0  # Binary label (COTS present or not)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (224, 224))  # Resize for CNN
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Define CNN Model
class COTSClassifier(nn.Module):
    def __init__(self):
        super(COTSClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Binary classification (COTS vs. No COTS)

    def forward(self, x):
        return self.model(x)

# Training Function
def train_cots_classifier(num_epochs=1, batch_size=16, learning_rate=0.001):
    print("Starting CNN Training for COTS Detection...")

    # Data Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load train dataset
    train_dataset = COTSDataset(TRAIN_CSV, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load test dataset
    test_dataset = COTSDataset(TEST_CSV, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COTSClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in tqdm(range(num_epochs), desc="Training Progress", leave=True):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"Model saved to {OUTPUT_MODEL_PATH}")

    # Evaluate model on test and save results
    metrics = evaluate_classifier(model, test_loader, device)

    with open(PERFORMANCE_OUTPUT_PATH, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"Model performance for test saved in {PERFORMANCE_OUTPUT_PATH}")

if __name__ == "__main__":
    train_cots_classifier()

