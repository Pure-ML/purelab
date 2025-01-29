import random
import numpy as np
from cleanlab import Datalab
from purelab.utils.data_checker_utils import *

def display_label_issues(lab, df, labels):
    """Display detailed information about label issues."""
    print("\n=== Label Issues Analysis (Regression) ===")
    label_results = lab.get_issues("label")
    
    total_issues = len(label_results[label_results['is_label_issue']])
    print(f"\nFound {total_issues} potential label issues")
    
    if total_issues > 0:
        # Set pandas display options to show all data
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # First show summary of label issues
        issues = label_results[label_results['is_label_issue']].sort_values('label_score')
        print("\nLabel issues summary:")
        print(issues[['is_label_issue', 'label_score']])
        
        print("\nDetailed label issues data:")
        # Then show the actual data for each issue
        issue_indices = issues.index
        display_df = df.iloc[issue_indices].copy()
        display_df['target_value'] = labels.iloc[issue_indices]
        display_df['label_score'] = label_results['label_score'].iloc[issue_indices]
        print(display_df)
        
        # Ask user about saving results
        save_option = input("\nWould you like to save the results to JSON? (y/n): ").strip().lower()
        if save_option == 'y':
            custom_path = input("Enter custom file path (or press Enter for default): ").strip()
            output_path = custom_path if custom_path else None
            save_results_to_json(label_results, output_path=output_path, issue_type="label")
        
        # Reset display options
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')

def main():
    try:
        print("\n=== Label Issues Checker (Regression) ===")
        
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
        display_label_issues(lab, df, labels)
        
        # Apply fixes if user wants
        df_fixed = apply_fixes(df, lab.get_issues(), "label")
        
        # Save fixed dataset if changes were made
        if not df_fixed.equals(df):
            save_fixed_dataset(df_fixed, None)
            
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