# VehicleAdvertAI - Machine Learning Services

This module contains the machine learning components of the VehicleAdvertAI project. It includes models for vehicle price prediction, fault detection from text, and image-based classification.

The system is designed to work as a separate Python service and is integrated with the .NET backend via HTTP APIs.

---

## Overview

This module provides three core machine learning capabilities:

1. Vehicle price prediction (regression)
2. Fault detection from user text input (NLP classification)
3. Image-based fault detection (computer vision)

Each model is trained separately and saved for later use in production.

---

## Technologies Used

- Python
- PyTorch (for image classification)
- Scikit-learn (for NLP and regression models)
- Pandas (data processing)
- Joblib (model persistence)

---

## Project Structure

src/
  train_car_price_model.py
  train_text_model.py
  train_image_model.py

data/
  car_data.csv
  car_faults.csv
  image_dataset/

models/
  car_price_model.pkl
  fault_model.pkl
  vectorizer.pkl
  image_model.pth


## 1. Vehicle Price Prediction Model

This model predicts vehicle prices based on structured input data.

### Features Used

- Brand
- Model
- Year
- Km
- GearType
- FuelType
- City

### Approach

- OneHotEncoder for categorical features
- RandomForestRegressor for prediction
- Pipeline used to combine preprocessing and model

### Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)

### Output

The trained model is saved as:

car_price_model.pkl

---

## 2. Text-Based Fault Detection Model

This model classifies vehicle issues based on user-provided descriptions.

### Approach

- TF-IDF vectorization (unigram + bigram)
- Logistic Regression classifier

### Example Input

"I hear a strange noise from the engine when accelerating"

### Output

Predicted issue label (e.g., engine_problem, brake_issue)

### Evaluation Metrics

- Accuracy
- Precision / Recall / F1 Score

### Output Files

- fault_model.pkl
- vectorizer.pkl

---

## 3. Image-Based Fault Detection Model

This model classifies vehicle faults from images using deep learning.

### Approach

- Transfer Learning with ResNet18 (pretrained on ImageNet)
- Final layer customized for dataset classes
- Trained using PyTorch

### Dataset Structure

image_dataset/
normal/
kablo_cikmis/
hortum_delik/


### Training Details

- Input size: 224x224
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Epochs: configurable (default: 5)

### Output

The trained model is saved as:

image_model.pth

---

## How to Run

### 1. Install dependencies

pip install torch torchvision scikit-learn pandas joblib

---

### 2. Train models

Run each script separately:

python train_car_price_model.py  
python train_text_model.py  
python train_image_model.py  

---

## Integration

- Models are consumed by a Flask API
- .NET backend communicates via HTTP requests
- Predictions are returned as JSON responses

---

## Notes

- Ensure datasets are correctly formatted before training
- Model performance depends heavily on dataset quality
- Image model requires a GPU for faster training (optional)

---

## Future Improvements

- Hyperparameter optimization
- Larger and more balanced datasets
- Model versioning
- Real-time inference optimization
- Dockerization for deployment

---

## Purpose

This module is developed as part of a full-stack AI-powered system to demonstrate practical machine learning integration in backend applications.

---

