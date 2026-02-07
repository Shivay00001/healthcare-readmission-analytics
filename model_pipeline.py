import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os

# Set style for plots
plt.style.use('ggplot')

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_pipeline(df):
    print("Preprocessing data...")
    
    # Drop non-predictive columns
    drop_cols = ['patient_id']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Define features and target
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"Categorical columns: {list(categorical_cols)}")
    print(f"Numerical columns: {list(numerical_cols)}")
    
    # Create preprocessing pipeline
    # Handle unknown categories by ignoring them during transform
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return X, y, preprocessor

def train_model(X, y, preprocessor):
    print("Training Random Forest model...")
    
    # Create full pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=100, 
                                                                  random_state=42, 
                                                                  class_weight='balanced'))])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train
    clf.fit(X_train, y_train)
    print("Model training complete.")
    
    return clf, X_test, y_test

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # ROC-AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature Importance (if model supports it)
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        
        # Get feature names from preprocessor
        # Logic to extract feature names after OneHotEncoding
        preprocessor = model.named_steps['preprocessor']
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_names = preprocessor.named_transformers_['num'].get_feature_names_out() # Or just numerical_cols
        
        feature_names = np.r_[num_names, cat_names]
        
        # Create DataFrame
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
        plt.title('Top 20 Feature Important Factors for Readmission')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved.")

    # Save metrics to file
    with open("model_metrics.txt", "w") as f:
        f.write(f"ROC-AUC Score: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

def main():
    data_path = "healthcare_dataset.csv"
    if not os.path.exists(data_path):
        print("Dataset not found. Please run data_generator.py first.")
        return

    df = load_data(data_path)
    X, y, preprocessor = preprocess_pipeline(df)
    model, X_test, y_test = train_model(X, y, preprocessor)
    evaluate_model(model, X_test, y_test)
    
    # Save model
    joblib.dump(model, 'readmission_model.joblib')
    print("Model saved to readmission_model.joblib")

if __name__ == "__main__":
    main()
