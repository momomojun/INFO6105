# AI Agent Instructions for Data Science Project

## Project Overview
This is a data science and machine learning workspace focusing on semi-supervised learning, classification, and model evaluation. The project includes multiple Jupyter notebooks implementing various ML algorithms and techniques.

## Key Project Components

### Data Processing Patterns
- Data preprocessing follows a consistent pattern across notebooks:
  1. Load data using `pd.read_csv()` from the `Datasets/` directory
  2. Handle missing values with median imputation
  3. Remove outliers using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
  4. Normalize numeric columns using min-max scaling
  5. Encode categorical variables using one-hot encoding

Example from `python_project.ipynb`:
```python
data = pd.read_csv('diabetes_project.csv')
# Handle missing values and outliers 
# Scale numeric features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_data)
```

### Model Training Workflow
1. Data splitting: Always use 80-20 train-test split with random_state=42
2. Feature engineering: PCA/LDA for dimensionality reduction
3. Model evaluation: Use 5-fold cross-validation for hyperparameter tuning
4. Super learner approach: Combine base models (Naïve Bayes, Neural Network, KNN) with a meta-learner

### Common Dependencies
Required packages are consistently used across notebooks:
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### File Structure
- `*.ipynb` files: Main analysis notebooks
- `Datasets/*.csv`: Source datasets
- `*_result.csv`: Generated output files
- Models expect data in CSV format with standard sklearn-compatible numeric features

### Best Practices
1. Always verify data quality before modeling:
   ```python
   data.info()
   data.isnull().sum()
   ```
2. Use StandardScaler before PCA/LDA
3. Save processed datasets for reproducibility
4. Document hyperparameter search results

## Integration Points
- Models trained on primary datasets (e.g., diabetes) are deployed on secondary datasets in `Datasets/`
- Feature engineering steps (PCA/LDA) must be consistent between training and inference
- Model outputs are saved as CSV files for further analysis

## Common Pitfalls
1. DataFrame copy warnings - use .copy() when creating data subsets
2. Missing value handling must be consistent between train and test
3. Feature scaling must be fitted on training data only
4. Check for data leakage when preprocessing