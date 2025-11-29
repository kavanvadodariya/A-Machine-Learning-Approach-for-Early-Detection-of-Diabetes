# Install the necessary library for tabular output (required in Colab if not already done)
!pip install tabulate

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tabulate import tabulate
import os
import sys

# --- Data Loading ---
file_name = "preprocessed_diabetes_dataset.csv" 
file_path = f"/content/{file_name}" 

# Load preprocessed data
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit()

# Split features and target
X = df.drop("Target", axis=1)
y = df["Target"]

# 70-30 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- KNN Optimization using Grid Search and Scaling ---

# 1. Create a Pipeline: This forces data scaling before applying KNN, which is essential.
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('knn', KNeighborsClassifier())
])

# 2. Define the parameter grid for K (the number of neighbors, from 1 to 20)
param_grid = {
    'knn__n_neighbors': range(1, 21)
}

# 3. Initialize GridSearchCV to find the best K using 5-fold cross-validation, optimizing for F1-score.
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1
)

# 4. Fit the GridSearch to the training data
print("Starting Grid Search to find optimal K...")
grid_search.fit(X_train, y_train)

# 5. Extract the best model and best K
best_k = grid_search.best_params_['knn__n_neighbors']
best_model = grid_search.best_estimator_

# --- Evaluation of the Best Model ---

# Predictions on the test set using the best, optimized model
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Value": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_proba)
    ]
}

# Convert to DataFrame for tabular output
results_df = pd.DataFrame(results)

# Print the final single table output
print("\n" + "="*60)
print(f"Optimized KNN Results: Best K Found = {best_k}")
print("="*60)
print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))
