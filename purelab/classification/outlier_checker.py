import random
import numpy as np
from cleanlab import Datalab
from purelab.utils.data_checker_utils import *

def display_outlier_issues(lab, df, labels):
    """Display detailed information about outliers."""
    print("\n=== Outlier Analysis (Classification) ===")
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
        display_df = df.iloc[outlier_indices].assign(
            given_label=labels.iloc[outlier_indices],
            outlier_score=outlier_results['outlier_score'].iloc[outlier_indices]
        )
        print(display_df)
        
        # Ask user about saving results
        save_option = input("\nWould you like to save the results to JSON? (y/n): ").strip().lower()
        if save_option == 'y':
            custom_path = input("Enter custom file path (or press Enter for default): ").strip()
            output_path = custom_path if custom_path else None
            save_results_to_json(outlier_results, output_path=output_path, issue_type="outlier")
        
        # Reset display options
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')

def main():
    try:
        print("\n=== Outlier Checker (Classification) ===")
        
        # Load and preprocess data
        df, label_column = load_data(task='classification')
        print("Data loaded successfully")  # Debug print
        
        # Let user select features
        feature_columns = get_feature_columns(df, label_column)
        print(f"\nSelected features: {', '.join(feature_columns)}")
        
        # Preprocess data
        print("\nPreprocessing data...")
        X_processed, labels = preprocess_data(df, feature_columns, label_column)
        print("Preprocessing complete")  # Debug print
        
        # Get prediction probabilities
        print("Computing out-of-sample predictions...")
        pred_probs = get_pred_probs(X_processed, labels, task='classification')
        print("Predictions complete")  # Debug print
        
        # Create KNN graph
        print("Creating KNN graph...")
        knn_graph = create_knn_graph(X_processed)
        print("KNN graph created")  # Debug print
        
        # Initialize and run Datalab
        print("Running Datalab analysis...")
        lab = Datalab(data=df, label_name=label_column)
        lab.find_issues(pred_probs=pred_probs, knn_graph=knn_graph)
        
        # Display results
        display_outlier_issues(lab, df, labels)
        
        # Apply fixes if user wants
        df_fixed = apply_fixes(df, lab.get_issues(), "outlier")
        
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