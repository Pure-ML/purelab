import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

ISSUE_TYPE_MAPPING = {
    'label': {
        'classification': 'label_classification',
        'regression': 'label_regression'
    },
    'outlier': {
        'classification': 'outlier_classification',
        'regression': 'outlier_regression'
    },
    'duplicate': {
        'classification': 'duplicate_classification',
        'regression': 'duplicate_regression'
    },
    'non_iid': {
        'classification': 'non_iid_classification',
        'regression': 'non_iid_regression'
    }
}

def get_feature_columns(df, label_column):
    """Let user select feature columns and automatically detect their types."""
    print("\nAvailable columns for features:")
    columns = [col for col in df.columns if col != label_column]
    for idx, col in enumerate(columns, 1):
        dtype = df[col].dtype
        print(f"{idx}. {col} ({dtype})")
    
    while True:
        feature_cols = input("\nEnter column numbers to use as features (comma-separated, or 'all'): ").strip()
        if feature_cols.lower() == 'all':
            return columns
        
        try:
            selected_indices = [int(i.strip()) - 1 for i in feature_cols.split(',')]
            selected_columns = [columns[i] for i in selected_indices]
            return selected_columns
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid column numbers or 'all'")

def preprocess_data(df, feature_columns, label_column, task='classification'):
    """Preprocess the data based on column types and task type."""
    # Extract features and labels
    X_raw = df[feature_columns].copy()
    labels = df[label_column].copy()
    
    # Automatically detect numeric and categorical columns
    numeric_features = X_raw.select_dtypes(include=[np.number]).columns
    categorical_features = X_raw.select_dtypes(exclude=[np.number]).columns
    
    if task == 'classification':
        # One-hot encode categorical features if any exist
        if len(categorical_features) > 0:
            print(f"\nOne-hot encoding categorical features: {', '.join(categorical_features)}")
            X_encoded = pd.get_dummies(X_raw, columns=categorical_features)
        else:
            X_encoded = X_raw.copy()
        
        # Standardize numeric features
        if len(numeric_features) > 0:
            print(f"\nStandardizing numeric features: {', '.join(numeric_features)}")
            scaler = StandardScaler()
            X_processed = X_encoded.copy()
            X_processed[numeric_features] = scaler.fit_transform(X_encoded[numeric_features])
            return X_processed, labels
        return X_encoded, labels
        
    else:  # regression
        print("\nHandling missing values...")
        
        # For numeric columns, fill NaN with median
        if len(numeric_features) > 0:
            for col in numeric_features:
                if X_raw[col].isnull().any():
                    median_val = X_raw[col].median()
                    X_raw[col] = X_raw[col].fillna(median_val)
                    print(f"Filled missing values in {col} with median: {median_val}")
        
        # For categorical columns, fill NaN with mode
        if len(categorical_features) > 0:
            for col in categorical_features:
                if X_raw[col].isnull().any():
                    mode_val = X_raw[col].mode()[0]
                    X_raw[col] = X_raw[col].fillna(mode_val)
                    print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # One-hot encode categorical features if any exist
        if len(categorical_features) > 0:
            print(f"\nOne-hot encoding categorical features: {', '.join(categorical_features)}")
            X_encoded = pd.get_dummies(X_raw, columns=categorical_features)
        else:
            X_encoded = X_raw.copy()
        
        # Standardize numeric features
        if len(numeric_features) > 0:
            print(f"\nStandardizing numeric features: {', '.join(numeric_features)}")
            scaler = StandardScaler()
            X_processed = X_encoded.copy()
            X_processed[numeric_features] = scaler.fit_transform(X_encoded[numeric_features])
            return X_processed, labels
        return X_encoded, labels

def get_pred_probs(X, y, task='classification', n_folds=5):
    """Get out-of-sample predictions using cross-validation."""
    if task == 'classification':
        clf = HistGradientBoostingClassifier(random_state=100)
        pred_probs = cross_val_predict(
            clf,
            X,
            y,
            cv=n_folds,
            method="predict_proba"
        )
    else:  # regression
        clf = HistGradientBoostingRegressor(random_state=100)
        pred_probs = cross_val_predict(
            clf,
            X,
            y,
            cv=n_folds
        )
    return pred_probs

def create_knn_graph(X):
    """Create KNN graph for better outlier detection."""
    KNN = NearestNeighbors(metric='euclidean')
    KNN.fit(X.values)
    return KNN.kneighbors_graph(mode="distance")

def load_data(task='classification'):
    """Load and validate the dataset."""
    while True:
        csv_path = input("\nEnter the path to your CSV file: ").strip()
        try:
            df = pd.read_csv(csv_path)
            print(f"\nDataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
            break
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            print("Please try again.")
    
    # Get label column
    print("\nAvailable columns:")
    for idx, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        print(f"{idx}. {col} ({dtype})")
    
    while True:
        label_column = input("\nEnter the name of the label column: ").strip()
        if label_column in df.columns:
            if task == 'classification':
                # Verify the label column has categorical/discrete values
                unique_labels = df[label_column].nunique()
                print(f"\nFound {unique_labels} unique classes in the label column")
                if unique_labels < 2:
                    print("Warning: Label column must have at least 2 classes")
                    continue
            else:  # regression
                # Verify the label column has numeric values
                if not np.issubdtype(df[label_column].dtype, np.number):
                    print("Warning: Label column must contain numeric values for regression")
                    continue
            break
        print(f"Error: Column '{label_column}' not found in the dataset.")
    
    return df, label_column 

def save_results_to_json(results: Dict, 
                        dataset_name: str, 
                        output_path: Optional[str] = None,
                        issue_type: str = None) -> None:
    """Save analysis results to a JSON file."""
    if output_path is None:
        os.makedirs('results', exist_ok=True)
        os.makedirs(f'results/{dataset_name}', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'results/{dataset_name}/{issue_type}_{timestamp}.json'
    
    try:
        # Convert DataFrame to records if needed
        if hasattr(results, 'to_dict'):
            issues = results.to_dict('records')
        else:
            issues = results

        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        # Convert results to serializable format
        serializable_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_name': dataset_name,
            'issue_type': issue_type,
            'issues': convert_numpy(issues)
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()

def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif pd.isna(obj):
        return None
    return obj 