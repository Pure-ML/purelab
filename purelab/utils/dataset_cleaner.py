import pandas as pd
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from glob import glob

class DatasetCleaner:
    """Class to clean datasets based on PureLab analysis results."""
    
    def __init__(self, original_dataset_path: str, results_dir: str):
        """Initialize the DatasetCleaner.
        
        Args:
            original_dataset_path: Path to the original CSV dataset
            results_dir: Directory containing JSON results from PureLab analysis
        """
        self.original_dataset_path = original_dataset_path
        self.results_dir = results_dir
        self.changes_log: List[str] = []
        
        try:
            self.dataset = pd.read_csv(original_dataset_path)
            self.changes_log.append(f"Loaded original dataset from {original_dataset_path}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")
    
    def _load_json_results(self, issue_type: str) -> Optional[Dict]:
        """Load the most recent JSON results for a given issue type.
        
        Args:
            issue_type: Type of issue (label, outlier, duplicate, non_iid)
            
        Returns:
            Dictionary containing the results or None if no results found
        """
        try:
            # Find all matching JSON files
            pattern = os.path.join(self.results_dir, f"{issue_type}_*.json")
            files = glob(pattern)
            
            if not files:
                self.changes_log.append(f"No results found for {issue_type}")
                return None
            
            # Get most recent file
            latest_file = max(files, key=os.path.getctime)
            
            with open(latest_file, 'r') as f:
                results = json.load(f)
            self.changes_log.append(f"Loaded results from {latest_file}")
            return results
            
        except Exception as e:
            self.changes_log.append(f"Error loading {issue_type} results: {str(e)}")
            return None
    
    def apply_label_corrections(self, task: str, label_column: str) -> None:
        """Apply label corrections based on label issues results."""
        issue_type = f"label_{task}"
        results = self._load_json_results(issue_type)
        
        if not results or not results.get('issues'):
            self.changes_log.append("No label corrections to apply")
            return
        
        corrections = 0
        for issue in results['issues']:
            # Try to find index and new label in various possible locations
            idx = None
            new_label = None
            
            # Look for index in common locations
            for idx_key in ['index', 'row_index']:
                idx = issue.get(idx_key) or issue.get('row_data', {}).get(idx_key)
                if idx is not None:
                    break
            
            # Look for label issue flag and predicted label
            is_label_issue = issue.get('is_label_issue') or issue.get('row_data', {}).get('is_label_issue')
            if is_label_issue:
                new_label = (issue.get('predicted_label') or 
                            issue.get('row_data', {}).get('predicted_label'))
            
            if new_label is not None and idx is not None:
                old_label = self.dataset.loc[idx, label_column]
                self.dataset.loc[idx, label_column] = new_label
                corrections += 1
                self.changes_log.append(
                    f"Corrected label at index {idx}: {old_label} -> {new_label}"
                )
        
        self.changes_log.append(f"Applied {corrections} label corrections")
    
    def remove_outliers(self) -> None:
        """Remove identified outliers from the dataset."""
        results = self._load_json_results('outlier')
        
        if not results or not results.get('issues'):
            self.changes_log.append("No outliers to remove")
            return
        
        outlier_indices = []
        for issue in results['issues']:
            # Check if it's an outlier directly from the issue dict
            if issue.get('is_outlier_issue', False):
                idx = issue.get('index')
                if idx is not None:
                    outlier_indices.append(idx)
        
        if outlier_indices:
            self.dataset.drop(outlier_indices, inplace=True)
            self.changes_log.append(f"Removed {len(outlier_indices)} outliers")
    
    def handle_duplicates(self) -> None:
        """Remove duplicate entries keeping the first instance."""
        results = self._load_json_results('duplicate')
        
        if not results or not results.get('issues'):
            self.changes_log.append("No duplicates to handle")
            return
        
        removed = 0
        for issue in results['issues']:
            if issue.get('is_near_duplicate_issue'):
                # Get duplicates either from direct field or nested row_data
                duplicates = (issue.get('near_duplicate_sets') or 
                             issue.get('row_data', {}).get('near_duplicate_sets'))
                
                if duplicates:
                    # Handle both single indices and lists of indices
                    if isinstance(duplicates[0], (list, tuple)):
                        # Classification case - duplicates is a list of indices forming a set
                        # Keep the first index, remove the rest
                        if len(duplicates) > 1:
                            # Filter out indices that don't exist in the dataset
                            indices_to_remove = [idx for idx in duplicates[1:] 
                                               if idx in self.dataset.index]
                            if indices_to_remove:
                                self.dataset.drop(indices_to_remove, inplace=True)
                                removed += len(indices_to_remove)
                                self.changes_log.append(
                                    f"Removed {len(indices_to_remove)} duplicates of index {duplicates[0]}"
                                )
                    else:
                        # Regression case - duplicates is a list of indices
                        # Remove all but the first index
                        if len(duplicates) > 1:
                            # Filter out indices that don't exist in the dataset
                            indices_to_remove = [idx for idx in duplicates[1:] 
                                               if idx in self.dataset.index]
                            if indices_to_remove:
                                self.dataset.drop(indices_to_remove, inplace=True)
                                removed += len(indices_to_remove)
                                self.changes_log.append(
                                    f"Removed {len(indices_to_remove)} duplicates of index {duplicates[0]}"
                                )
        
        if removed > 0:
            self.changes_log.append(f"Removed {removed} total duplicate entries")
    
    def remove_non_iid(self) -> None:
        """Remove identified non-IID samples."""
        results = self._load_json_results('non_iid')
        
        if not results or not results.get('issues'):
            self.changes_log.append("No non-IID samples to remove")
            return
        
        non_iid_indices = []
        for issue in results['issues']:
            # Check both direct and nested row_data for non-IID flag
            is_non_iid = (issue.get('is_non_iid_issue') or 
                          issue.get('row_data', {}).get('is_non_iid_issue'))
            if is_non_iid:
                idx = issue.get('index')
                if idx is not None:
                    non_iid_indices.append(idx)
        
        if non_iid_indices:
            self.dataset.drop(non_iid_indices, inplace=True)
            self.changes_log.append(f"Removed {len(non_iid_indices)} non-IID samples")
    
    def clean_dataset(self, task: str, output_path: str, label_column: str) -> None:
        """Apply all cleaning operations and save the results.
        
        Args:
            task: Either 'classification' or 'regression'
            output_path: Path to save the cleaned dataset
            label_column: Name of the label column in the dataset
        """
        if task not in ['classification', 'regression']:
            raise ValueError("Task must be either 'classification' or 'regression'")
        
        if label_column not in self.dataset.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        self.changes_log.append("\nStarting dataset cleaning process...")
        original_rows = len(self.dataset)
        
        # Apply corrections in order
        self.apply_label_corrections(task, label_column)
        self.remove_outliers()
        self.handle_duplicates()
        self.remove_non_iid()
        
        # Save cleaned dataset
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.dataset.to_csv(output_path, index=False)
        final_rows = len(self.dataset)
        self.changes_log.append(f"\nCleaning complete:")
        self.changes_log.append(f"Original rows: {original_rows}")
        self.changes_log.append(f"Final rows: {final_rows}")
        self.changes_log.append(f"Rows removed: {original_rows - final_rows}")
        self.changes_log.append(f"\nCleaned dataset saved to: {output_path}")
        
        # Save changes log
        log_path = output_path.rsplit('.', 1)[0] + '_changes.txt'
        with open(log_path, 'w') as f:
            f.write('\n'.join(self.changes_log))
        print(f"Changes log saved to: {log_path}") 