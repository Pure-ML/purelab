import random
import numpy as np
from cleanlab import Datalab
from purelab.utils.data_checker_utils import *

def display_outlier_issues(lab, df, labels):
    """Display detailed information about outliers."""
    print("\n=== Outlier Analysis (Regression) ===")
    outlier_results = lab.get_issues("outlier")
    
    total_outliers = len(outlier_results[outlier_results['is_outlier_issue']])
    print(f"\nFound {total_outliers} potential outliers")
    
    if total_outliers > 0:
        # Set pandas display options to show all data
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # First show summary of outliers
        outliers = outlier_results[outlier_results['is_outlier_issue']].sort_values('outlier_score')
        print("\nOutlier summary:")
        print(outliers[['is_outlier_issue', 'outlier_score']])
        
        print("\nDetailed outlier data:")
        # Then show the actual data for each outlier
        outlier_indices = outliers.index
        display_df = df.iloc[outlier_indices].copy()
        display_df['target_value'] = labels.iloc[outlier_indices]
        display_df['outlier_score'] = outlier_results['outlier_score'].iloc[outlier_indices]
        print(display_df)
        
        # Reset display options
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')

    # Ask user about saving results (moved outside the if block)
    save_option = input("\nWould you like to save the results to JSON? (y/n): ").strip().lower()
    if save_option == 'y':
        # Get dataset name from the user
        dataset_name = input("Enter dataset name: ").strip()
        custom_path = input("Enter custom file path (or press Enter for default): ").strip()
        output_path = custom_path if custom_path else None
        save_results_to_json(outlier_results, dataset_name, output_path=output_path, issue_type="outlier_regression")

def main():
    try:
        print("\n=== Outlier Checker (Regression) ===")
        
        # Load and preprocess data
        df, label_column = load_data(task='regression')
        print("Data loaded successfully")  # Debug print
        
        feature_columns = get_feature_columns(df, label_column)
        print(f"\nSelected features: {', '.join(feature_columns)}")
        
        # Preprocess data
        print("\nPreprocessing data...")
        X_processed, labels = preprocess_data(df, feature_columns, label_column, task='regression')
        print("Preprocessing complete")  # Debug print
        
        # Get predictions
        print("Computing out-of-sample predictions...")
        predictions = get_pred_probs(X_processed, labels, task='regression')
        print("Predictions complete")  # Debug print
        
        # Create KNN graph
        print("Creating KNN graph...")
        knn_graph = create_knn_graph(X_processed)
        print("KNN graph created")  # Debug print
        
        # Initialize and run Datalab
        print("Running Datalab analysis...")
        lab = Datalab(data=df, label_name=label_column, task="regression")
        lab.find_issues(pred_probs=predictions, knn_graph=knn_graph)
        
        # Display results
        display_outlier_issues(lab, df, labels)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    np.random.seed(100)
    random.seed(100)
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 