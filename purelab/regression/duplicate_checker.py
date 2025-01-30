import random
import numpy as np
from cleanlab import Datalab
from purelab.utils.data_checker_utils import *

def display_duplicate_issues(lab, df, labels):
    """Display detailed information about near-duplicates for regression."""
    print("\n=== Near-Duplicate Analysis (Regression) ===")
    duplicate_results = lab.get_issues("near_duplicate")
    
    total_duplicates = len(duplicate_results[duplicate_results['is_near_duplicate_issue']])
    print(f"\nFound {total_duplicates} potential near-duplicates")
    
    if total_duplicates > 0:
        # Set pandas display options to show all data
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # First show summary of duplicates
        duplicates = duplicate_results[duplicate_results['is_near_duplicate_issue']].sort_values('near_duplicate_score')
        print("\nDuplicate summary:")
        print(duplicates[['is_near_duplicate_issue', 'near_duplicate_score', 'near_duplicate_sets', 'distance_to_nearest_neighbor']])
        
        print("\nDetailed duplicate sets:")
        # Then show the actual data for each duplicate set
        for idx in duplicates.index:
            indices_to_display = [idx]
            if isinstance(duplicate_results.loc[idx, "near_duplicate_sets"], list):
                indices_to_display.extend(duplicate_results.loc[idx, "near_duplicate_sets"])
            
            # Display all rows in the set together
            display_df = df.iloc[indices_to_display].copy()
            display_df['target_value'] = labels.iloc[indices_to_display]
            print(display_df)
            print("-" * 80)
        
        # Ask user about saving results
        save_option = input("\nWould you like to save the results to JSON? (y/n): ").strip().lower()
        if save_option == 'y':
            # Get dataset name from the user
            dataset_name = input("Enter dataset name: ").strip()
            custom_path = input("Enter custom file path (or press Enter for default): ").strip()
            output_path = custom_path if custom_path else None
            save_results_to_json(duplicate_results, dataset_name, output_path=output_path, issue_type="duplicate_regression")
        
        # Reset display options
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')

def main():
    try:
        print("\n=== Duplicate Checker (Regression) ===")
        
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
        display_duplicate_issues(lab, df, labels)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    print("Script starting...")  # Debug print
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