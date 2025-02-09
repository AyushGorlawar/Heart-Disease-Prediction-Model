import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
import io

def download_data():
    """
    Downloads the heart disease dataset and saves it locally
    """
   
    url = "https://storage.googleapis.com/kaggle-data-sets/529/982/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240209%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240209T110830Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=708e02d0c669115d75fb57c4a279d3c5094a20603112523aab069d417e1d456505391146c0eae5ce90de46923636462556a694147f58cc1c39c59a5041df73f86a1e648a2a785d8399cd44c1fe1b1515266b5fd885b7601f601d9b82c0143eb95f62903f677aa74ce707e7b2f7dd67a89e7ca641dba31986c01ebf611cd6a3fba55a9467d68d1d8aa654b84a4dfc9a7f0ad0c7ea17b3e8d419a6d6f8f2c0ea97f3eae66f5f232e04af76f5de37c5833db855aa65af47fc370e2ad0f44fb4859e0a81e582f01f743cdf2d4f894a994f7a3c56d9fd8df2e436ec5dde0843034f3f614b67b34c9001877fec02af2b83aee8b3083c5ffad66cc9775c"
    
    try:
        print("Downloading dataset...")
        response = requests.get(url)
        response.raise_for_status()
        
  
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Save the file
        with open('data/heart.csv', 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nUsing backup data...")
        create_backup_data()

def create_backup_data():
    """
    Creates a backup dataset with 100 samples if download fails
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(150, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(100, 220, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples).round(1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 3, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    pd.DataFrame(data).to_csv('data/heart.csv', index=False)
    print("Created backup dataset with 100 samples.")

def load_and_prepare_data():
    """
    Loads and prepares the data for training
    """
    try:
        # Try to load the data
        if not os.path.exists('data/heart.csv'):
            download_data()
        
        # Read the dataset
        data = pd.read_csv('data/heart.csv')
        print("\nDataset shape:", data.shape)
        print("\nFeatures in the dataset:", list(data.columns))
        
        # Basic data validation
        if len(data) < 10: 
            print("Dataset too small, creating backup data...")
            create_backup_data()
            data = pd.read_csv('data/heart.csv')
            
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """
    Preprocesses the data for training
    """
    try:
  
        X = data.drop('target', axis=1)
        y = data['target']
        
   
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
  
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains and evaluates the model
    """
    try:
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.2%}")
        
   
        n_splits = min(5, len(X_train) // 10) 
        if n_splits > 1:
            cv_scores = cross_val_score(model, X_train, y_train, cv=n_splits)
            print(f"Cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
     
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, accuracy
        
    except Exception as e:
        print(f"Error in model training/evaluation: {e}")
        raise

def plot_results(model, X_test, y_test, feature_names):
    """
    Creates visualizations of the results
    """
    try:
   
        plt.figure(figsize=(15, 6))
        
   
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
 
        plt.subplot(1, 2, 2)
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_imp)
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in plotting: {e}")
        print("Skipping visualizations...")

def save_model(model):
    """
    Saves the trained model
    """
    try:
        from joblib import dump
        if not os.path.exists('models'):
            os.makedirs('models')
        dump(model, 'models/heart_disease_model.joblib')
        print("\nModel saved successfully!")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        print("Skipping model saving...")

def main():
    """
    Main function to run the entire program
    """
    try:
        print("Starting Heart Disease Prediction Program...")
        print("=========================================")
        
        # Load and prepare data
        data = load_and_prepare_data()
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
        
        # Train model
        model, accuracy = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, feature_names
        )
        
        # Plot results
        plot_results(model, X_test, y_test, feature_names)
        save_model(model)
        
        print("\nProgram completed successfully!")
        return model, accuracy
        
    except Exception as e:
        print(f"\nProgram failed: {e}")
        print("Please check the error messages above and try again.")
        raise

if __name__ == "__main__":
    try:
        model, accuracy = main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nProgram terminated due to error: {e}")
