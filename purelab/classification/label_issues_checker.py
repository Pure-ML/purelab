import random
import numpy as np
from cleanlab import Datalab
from purelab.utils.data_checker_utils import *

def display_label_issues(lab, df, labels):
    """Display detailed information about label issues."""
    print("\n=== Label Issues Analysis (Classification) ===")
    
    # Get all issues
    all_issues = lab.get_issues()
    
    # Print summary
    print(f"\nDataset Information: num_examples: {len(df)}, num_classes: {len(labels.unique())}")
    print("\nHere is a summary of various issues found in your data:")
    
    # Count issues by type
    issue_counts = {
        'label': len(all_issues[all_issues['is_label_issue']]),
        'outlier': len(all_issues[all_issues['is_outlier_issue']]),
        'near_duplicate': len(all_issues[all_issues['is_near_duplicate_issue']]),
        'non_iid': len(all_issues[all_issues['is_non_iid_issue']])
    }
    
    # Print issue counts
    for issue_type, count in issue_counts.items():
        print(f"{issue_type:>15}: {count:>5}")
    
    # Get and display label issues
    label_results = lab.get_issues("label")
    
    total_issues = len(label_results[label_results['is_label_issue']])
    print(f"\nFound {total_issues} potential label issues")
    
    if total_issues > 0:
        print("\nAll suspicious labels (sorted by severity):")
        # Set pandas display options to show all columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        suspicious_labels = label_results[label_results['is_label_issue']].sort_values('label_score')
        print(df.iloc[suspicious_labels.index].assign(
            given_label=labels.iloc[suspicious_labels.index],
            predicted_label=suspicious_labels["predicted_label"]
        ))
        
        # Reset display options
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        
        # Ask user about saving results
        save_option = input("\nWould you like to save the results to JSON? (y/n): ").strip().lower()
        if save_option == 'y':
            custom_path = input("Enter custom file path (or press Enter for default): ").strip()
            output_path = custom_path if custom_path else None
            save_results_to_json(label_results, output_path=output_path, issue_type="label")

def main():
    try:
        print("\n=== Label Issues Checker (Classification) ===")
        
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