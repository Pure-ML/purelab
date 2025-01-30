# PureLab: Data Quality Analysis Tool

PureLab is a Python package built on top of Cleanlab that helps identify various data quality issues in both classification and regression datasets. It provides automated analysis tools to detect label issues, outliers, duplicates, and non-IID (Independent and Identically Distributed) data points.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/purelab.git
   cd purelab
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install individual packages:
   ```bash
   pip install cleanlab pandas numpy scikit-learn
   ```

3. **Install PureLab:**
   ```bash
   pip install -e .
   ```

## Features

### 1. Label Issues Checker
Identifies potential problems with data labels.

**Classification:**
- Detects mislabeled examples using confidence scores and out-of-distribution analysis
- Compares predicted vs. given labels to find inconsistencies
- Command: `python -m purelab.classification.label_issues_checker`

**Regression:**
- Identifies target values that deviate significantly from predicted values
- Uses statistical methods to detect anomalous label patterns
- Command: `python -m purelab.regression.label_issues_checker`

### 2. Outlier Checker
Detects data points that significantly deviate from the overall pattern.

**Classification:**
- Identifies samples that don't fit well with any class
- Uses distance-based and density-based methods
- Command: `python -m purelab.classification.outlier_checker`

**Regression:**
- Detects numerical outliers in both features and target values
- Considers multivariate relationships
- Command: `python -m purelab.regression.outlier_checker`

### 3. Duplicate Checker
Finds near-duplicate entries in the dataset.

**Classification:**
- Identifies samples with similar feature values but possibly different labels
- Uses KNN graph for similarity computation
- Command: `python -m purelab.classification.duplicate_checker`

**Regression:**
- Detects samples with similar features but potentially different target values
- Considers numerical proximity in feature space
- Command: `python -m purelab.regression.duplicate_checker`

### 4. Non-IID Checker
Identifies violations of the IID assumption.

**Classification:**
- Detects class imbalances and distribution shifts
- Analyzes feature distribution across classes
- Command: `python -m purelab.classification.non_iid_checker`

**Regression:**
- Identifies clusters with different target value distributions
- Analyzes feature-target relationships
- Command: `python -m purelab.regression.non_iid_checker`

## Usage

1. **Input Data:**
   - Data should be in CSV format
   - Must have both feature columns and a label/target column
   - Can handle both numerical and categorical features

2. **Running Tests:**
   ```bash
   # For classification tasks
   python -m purelab.classification.<checker_name>
   
   # For regression tasks
   python -m purelab.regression.<checker_name>
   ```

3. **Interactive Process:**
   - You'll be prompted to:
     1. Enter the path to your CSV file
     2. Specify the label/target column
     3. Select feature columns to use
   - The tool will then:
     1. Preprocess the data
     2. Run the analysis
     3. Display results
     4. Offer to save results

4. **Outputs:**
   - Console output showing detailed analysis
   - Option to save results as JSON files
   - Results are saved in `results/<dataset_name>/` directory
   - JSON files include:
     - Metadata about the analysis
     - Summary statistics
     - Detailed issue information
     - Original row data for problematic samples

## JSON Output Structure
```json
{
"metadata": {
"timestamp": "2024-01-30T10:44:40.370175",
"dataset": "example_dataset",
"issue_type": "label_classification"
},
"summary": {
"total_issues": 5
},
"issues": [
{
"index": 42,
"issue_score": 0.95,
"row_data": {
// Original data row
}
}
]
}
```

## Implementation Details

- Uses Cleanlab's Datalab for core analysis
- Implements custom preprocessing for different data types
- Handles missing values automatically
- Provides consistent interface across all checkers
- Supports both classification and regression tasks with task-specific adaptations

## Requirements
- Python 3.7+
- cleanlab
- pandas
- numpy
- scikit-learn

## Note
This tool is designed to assist in data quality analysis but should not be used as the sole decision-maker for data cleaning. Always review the identified issues in the context of your specific use case.

