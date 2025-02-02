import argparse
from purelab.utils.dataset_cleaner import DatasetCleaner

def main():
    parser = argparse.ArgumentParser(description='Clean dataset using PureLab results')
    parser.add_argument('--input', required=True, help='Path to original dataset CSV')
    parser.add_argument('--results-dir', required=True, help='Directory containing JSON results')
    parser.add_argument('--output', required=True, help='Path to save cleaned dataset')
    parser.add_argument('--task', required=True, choices=['classification', 'regression'], 
                       help='Task type (classification or regression)')
    parser.add_argument('--label-column', required=True,
                       help='Name of the label column in the dataset')
    
    args = parser.parse_args()
    
    cleaner = DatasetCleaner(
        original_dataset_path=args.input,
        results_dir=args.results_dir
    )
    
    cleaner.clean_dataset(
        task=args.task,
        output_path=args.output,
        label_column=args.label_column
    )

if __name__ == "__main__":
    main() 