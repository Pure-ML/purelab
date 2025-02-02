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
    
    def _load_json_results(self, issue_type: str, task: str) -> Optional[Dict]:
        """Load task-specific JSON results."""
        try:
            # Handle both patterns for duplicate files
            if issue_type == 'duplicate':
                pattern1 = os.path.join(self.results_dir, f"{issue_type}_{task}_*.json")
                pattern2 = os.path.join(self.results_dir, f"{issue_type}_*.json")
                files = glob(pattern1) + glob(pattern2)
            else:
                pattern = os.path.join(self.results_dir, f"{issue_type}_{task}_*.json")
                files = glob(pattern)
            
            if not files:
                self.changes_log.append(f"No {task} results found for {issue_type}")
                return None
            
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, 'r') as f:
                results = json.load(f)
            self.changes_log.append(f"Loaded results from {latest_file}")
            return results
            
        except Exception as e:
            self.changes_log.append(f"Error loading {issue_type} results: {str(e)}")
            return None
    
    def apply_label_corrections(self, task: str, label_column: str) -> None:
        """Apply label corrections based on task-specific format."""
        results = self._load_json_results('label', task)
        
        if not results or not results.get('issues'):
            self.changes_log.append("No label corrections to apply")
            return
        
        corrections = 0
        for i, issue in enumerate(results['issues']):
            if task == 'regression':
                idx = issue.get('index')
                is_label_issue = issue.get('row_data', {}).get('is_label_issue')
                new_label = issue.get('row_data', {}).get('predicted_label')
            else:  # classification
                idx = i  # Index in the results list corresponds to dataset index
                is_label_issue = issue.get('is_label_issue')
                new_label = issue.get('predicted_label')
            
            if is_label_issue and new_label is not None and idx is not None:
                try:
                    old_label = self.dataset.loc[idx, label_column]
                    self.dataset.loc[idx, label_column] = new_label
                    corrections += 1
                    self.changes_log.append(
                        f"Corrected label at index {idx}: {old_label} -> {new_label}"
                    )
                except KeyError:
                    self.changes_log.append(f"Warning: Could not find index {idx} in dataset")
                    continue
        
        self.changes_log.append(f"Applied {corrections} label corrections")
    
    def remove_outliers(self, task: str) -> None:
        """Remove outliers using task-specific format."""
        results = self._load_json_results('outlier', task)
        
        if not results or not results.get('issues'):
            self.changes_log.append("No outliers to remove")
            return
        
        outlier_indices = []
        for issue in results['issues']:
            if task == 'regression':
                is_outlier = issue.get('row_data', {}).get('is_outlier_issue')
                idx = issue.get('index')
            else:
                is_outlier = issue.get('is_outlier_issue')
                idx = issue.get('index')
            
            if is_outlier and idx is not None:
                outlier_indices.append(idx)
        
        if outlier_indices:
            self.dataset.drop(outlier_indices, inplace=True)
            self.changes_log.append(f"Removed {len(outlier_indices)} outliers")
    
    def handle_duplicates(self, task: str) -> None:
        """Handle duplicates using task-specific format."""
        results = self._load_json_results('duplicate', task)
        
        if not results or not results.get('issues'):
            self.changes_log.append("No duplicates to handle")
            return
        
        removed = 0
        if task == 'regression':
            # Track duplicate pairs (keeping the lower index as original)
            duplicate_pairs = set()  # Using set to avoid duplicates
            
            for i, issue in enumerate(results['issues']):
                if issue.get('is_near_duplicate_issue'):
                    duplicate_sets = issue.get('near_duplicate_sets', [])
                    if duplicate_sets:
                        for dup_idx in duplicate_sets:
                            # Always keep the lower index as original
                            original_idx = min(i, dup_idx)
                            duplicate_idx = max(i, dup_idx)
                            duplicate_pairs.add((original_idx, duplicate_idx))
            
            # Process duplicate pairs
            indices_to_remove = set()
            for original_idx, duplicate_idx in duplicate_pairs:
                self.changes_log.append(
                    f"Found duplicate pair: keeping {original_idx}, removing {duplicate_idx}"
                )
                indices_to_remove.add(duplicate_idx)
            
            # Remove duplicates (keeping originals)
            valid_indices = [idx for idx in indices_to_remove if idx in self.dataset.index]
            if valid_indices:
                self.dataset.drop(valid_indices, inplace=True)
                removed = len(valid_indices)
                self.changes_log.append(f"Removed {removed} duplicate entries")
        else:
            # Existing classification duplicate handling
            for issue in results['issues']:
                if issue.get('is_near_duplicate_issue'):
                    duplicate_sets = issue.get('near_duplicate_sets', [])
                    if duplicate_sets:
                        original_idx = duplicate_sets[0]
                        valid_indices = [idx for idx in duplicate_sets[1:] if idx in self.dataset.index]
                        if valid_indices:
                            self.dataset.drop(valid_indices, inplace=True)
                            removed += len(valid_indices)
                            self.changes_log.append(
                                f"Removed {len(valid_indices)} duplicates of index {original_idx}"
                            )
        
        if removed > 0:
            self.changes_log.append(f"Removed {removed} total duplicate entries")
        else:
            self.changes_log.append("No duplicates to remove")
    
    def remove_non_iid(self, task: str) -> None:
        """Remove non-IID samples using task-specific format."""
        results = self._load_json_results('non_iid', task)
        
        if not results or not results.get('issues'):
            self.changes_log.append("No non-IID samples to remove")
            return
        
        non_iid_indices = []
        for issue in results['issues']:
            if task == 'regression':
                is_non_iid = issue.get('row_data', {}).get('is_non_iid_issue')
                idx = issue.get('index')
            else:
                is_non_iid = issue.get('is_non_iid_issue')
                idx = issue.get('index')
            
            if is_non_iid and idx is not None:
                non_iid_indices.append(idx)
        
        if non_iid_indices:
            self.dataset.drop(non_iid_indices, inplace=True)
            self.changes_log.append(f"Removed {len(non_iid_indices)} non-IID samples")
    
    def clean_dataset(self, task: str, output_path: str, label_column: str) -> None:
        """Clean dataset with task-specific handling."""
        if task not in ['classification', 'regression']:
            raise ValueError("Task must be either 'classification' or 'regression'")
        
        if label_column not in self.dataset.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        self.changes_log.append("\nStarting dataset cleaning process...")
        original_rows = len(self.dataset)
        
        # Apply corrections in order with task parameter
        self.apply_label_corrections(task, label_column)
        self.remove_outliers(task)
        self.handle_duplicates(task)
        self.remove_non_iid(task)
        
        # Save cleaned dataset and log
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.dataset.to_csv(output_path, index=False)
        final_rows = len(self.dataset)
        
        self.changes_log.extend([
            "\nCleaning complete:",
            f"Original rows: {original_rows}",
            f"Final rows: {final_rows}",
            f"Rows removed: {original_rows - final_rows}",
            f"\nCleaned dataset saved to: {output_path}"
        ])
        
        # Save changes log
        log_path = output_path.rsplit('.', 1)[0] + '_changes.txt'
        with open(log_path, 'w') as f:
            f.write('\n'.join(self.changes_log))
        print(f"Changes log saved to: {log_path}") 