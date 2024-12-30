import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data_cleaned = data.drop(columns=['Nama User'])
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data_cleaned['tujuan_investasi'] = label_encoder.fit_transform(data_cleaned['tujuan_investasi'])
    
    # Convert boolean columns
    boolean_columns = ['menikah', 'pernah_investasi', 'investasi_jika_market_turun']
    for col in boolean_columns:
        data_cleaned[col] = data_cleaned[col].astype(int)
    
    return data_cleaned, label_encoder

# Perform balanced sampling
def perform_balanced_sampling(data, target_col):
    classes = data[target_col].unique()
    max_size = max(len(data[data[target_col] == c]) for c in classes)
    
    balanced_data = []
    for c in classes:
        class_data = data[data[target_col] == c]
        upsampled = resample(class_data, replace=True, n_samples=max_size, random_state=42)
        balanced_data.append(upsampled)
    
    return pd.concat(balanced_data)

# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Main training and evaluation function
def train_and_evaluate_model():
    # Load and preprocess data
    data_cleaned, label_encoder = load_and_preprocess_data('tipe_investor.csv')
    X = data_cleaned.drop(columns=['tipe_investor'])
    y = label_encoder.fit_transform(data_cleaned['tipe_investor'])
    
    # Balance data
    data_balanced = pd.concat([X, pd.DataFrame(y, columns=['tipe_investor'])], axis=1)
    data_balanced = perform_balanced_sampling(data_balanced, 'tipe_investor')
    X_balanced = data_balanced.drop(columns=['tipe_investor'])
    y_balanced = data_balanced['tipe_investor']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Grid search for hyperparameter tuning
    param_grid = {
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'n_neighbors': [1, 3, 5]
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Print best parameters and evaluate the model
    print("\nBest Parameters from Grid Search:")
    print(grid_search.best_params_)
    y_pred = best_model.predict(X_test_scaled)
    print(f"\nAccuracy with Best Model: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print("\nCross-Validation Results:")
    print(f"Mean Accuracy: {cv_scores.mean():.2f}")
    print(f"Standard Deviation: {cv_scores.std():.2f}")
    
    # Save the best model and preprocessing tools
    joblib.dump(best_model, 'knn_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nModel, encoder, and scaler saved successfully.")

if __name__ == "__main__":
    train_and_evaluate_model()
