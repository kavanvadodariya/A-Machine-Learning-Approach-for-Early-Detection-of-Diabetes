#SVM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tabulate import tabulate

# File path of preprocessed dataset
file_path = "/content/preprocessed_diabetes_dataset.csv"

# Load preprocessed data
df = pd.read_csv(file_path)

# Split features and target
X = df.drop("Target", axis=1)
y = df["Target"]

# 70-30 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train SVM model
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

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

# Print table
print("SVM Model Evaluation Metrics:\n")
print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))
