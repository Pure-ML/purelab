import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
import json
import os

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

def save_results_to_json(results, dataset_name, output_path=None, issue_type=""):
    """Save analysis results to a JSON file."""
    try:
        if output_path is None:
            # Create results directory if it doesn't exist
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Create subdirectory for this dataset if it doesn't exist
            dataset_dir = os.path.join(results_dir, dataset_name)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            
            # Generate filename with timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(dataset_dir, f"{issue_type}_{timestamp}.json")
        
        # Map issue types to their Cleanlab column names
        issue_type_mapping = {
            "label_classification": "label",
            "label_regression": "label",
            "outlier_classification": "outlier",
            "outlier_regression": "outlier",
            "duplicate_classification": "near_duplicate",
            "duplicate_regression": "near_duplicate",
            "non_iid_classification": "non_iid",
            "non_iid_regression": "non_iid"
        }
        
        base_issue_type = issue_type_mapping.get(issue_type, issue_type.split('_')[0])
        issue_column = f"is_{base_issue_type}_issue"
        score_column = f"{base_issue_type}_score"
        
        # Convert results to JSON-serializable format
        json_results = {
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "dataset": dataset_name,
                "issue_type": issue_type
            },
            "summary": {
                "total_issues": len(results[results[issue_column]])
            },
            "issues": []
        }
        
        # Add detailed issue information
        for idx in results[results[issue_column]].index:
            # Get the row data
            row_data = results.loc[idx].to_dict()
            
            # Convert any numpy or pandas types to Python native types
            row_data = {k: convert_to_serializable(v) for k, v in row_data.items()}
            
            # Create the issue data dictionary
            issue_data = {
                "index": int(idx),
                f"{base_issue_type}_score": float(results.loc[idx, score_column]),
                "row_data": row_data
            }
            
            # Add near_duplicate_sets for duplicate issues
            if base_issue_type == "near_duplicate" and "near_duplicate_sets" in results.columns:
                sets = results.loc[idx, "near_duplicate_sets"]
                if isinstance(sets, list):
                    issue_data["near_duplicate_sets"] = [int(x) for x in sets]
            
            json_results["issues"].append(issue_data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError saving results to JSON: {str(e)}")

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