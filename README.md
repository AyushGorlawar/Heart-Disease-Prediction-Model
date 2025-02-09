# Heart Disease Prediction

## Overview
This project implements a **Heart Disease Prediction** system using **Random Forest Classifier**. The dataset is either downloaded or generated as a backup, and the model is trained to predict heart disease based on various medical attributes.

## Features
- Downloads the heart disease dataset automatically
- Creates a backup dataset if the download fails
- Preprocesses the dataset (scaling, splitting)
- Trains a **Random Forest Classifier**
- Evaluates the model using accuracy, cross-validation, and classification reports
- Generates visualizations (Confusion Matrix, Feature Importance)
- Saves the trained model for future use

## Installation
Ensure you have Python 3 installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `requests`
- `joblib`

## Usage
Run the program using:
```bash
python main.py
```
The program will:
1. Download or create the dataset
2. Preprocess the data
3. Train and evaluate the Random Forest model
4. Display results
5. Save the trained model

## Data Processing
- **Feature Scaling:** StandardScaler is used for normalization.
- **Splitting:** 80% Training, 20% Testing
- **Backup Data:** If dataset download fails, a synthetic dataset with 100 samples is generated.

## Model Training
- Model: **RandomForestClassifier**
- Parameters:
  - `n_estimators=100`
  - `max_depth=10`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
  - `random_state=42`

## Evaluation Metrics
- **Accuracy Score**
- **Cross-validation Score**
- **Confusion Matrix**
- **Classification Report**

## Visualizations
- **Confusion Matrix:** Displays model performance.
- **Feature Importance:** Identifies the most important features in prediction.

## Model Saving
The trained model is saved in the `models/` directory as:
```bash
models/heart_disease_model.joblib
```
## Output 

Starting Heart Disease Prediction Program...
=========================================

Dataset shape: (5, 14)

Model Accuracy: 80.00%
Cross-validation accuracy: 88.25% (+/- 35.36%)

Classification Report:
              precision    recall  f1-score   support

           0       0.43      0.75      0.55         8
           1       0.67      0.33      0.44        12

    accuracy                           0.80        60
   macro avg       0.55      0.54      0.49        20
weighted avg       0.57      0.50      0.48        20

![Project Screenshot](https://github.com/AyushGorlawar/Heart-Disease-Prediction-Model/blob/main/Output.png)

## Contribution
Feel free to contribute! Fork the repository, create a feature branch, and submit a pull request.

## License
This project is licensed under the MIT License.

---
**Author:** Ayush

