# Crown-of-Thorns Starfish Detection

This README is still a WIP!

## Overview
This project aims to develop an automated pipeline for detecting Crown-of-Thorns Starfish (COTS) in underwater images from the Great Barrier Reef. COTS outbreaks pose a significant threat to coral ecosystems, and early detection is crucial for conservation efforts.

Using machine learning, this project builds an **ensemble model** that:
1. **Classifies images** to determine whether COTS are present using a CNN.
2. **Detects and localizes** COTS using the YOLO object detection model.

The dataset comes from the [Kaggle TensorFlow Great Barrier Reef competition](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/overview), consisting of thousands of labeled underwater images.

## Repository Structure
They key directories to be aware of:
```
├── proposal/           # Project proposal and initial planning documents
├── eda/               # Exploratory Data Analysis (EDA) notebooks and findings
├── data_processing/   # Scripts for preparing data, train-test splitting, and preprocessing
├── model/             # Training scripts and saved models
│   ├── outputs/      # Model predictions, logs, and performance metrics
└── README.md          # Project documentation
```

## Sample Image
Below is an example of a COTS from an image in the dataset, where a red bounding box highlights the starfish:

![Underwater Image of COTS with Bounding Box](proposal/images/bounded_image.jpg)