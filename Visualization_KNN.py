import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd # Ensure pandas is imported if not already in the scope

# Note: results_df, y_test, y_pred, and y_proba must be available from the previous Optimized KNN run.

# Convert results to usable format
metric_names = results_df["Metric"].values
metric_values = results_df["Value"].values

## 1. Bar Chart of Performance Metrics
plt.figure(figsize=(8,5))
plt.bar(metric_names, metric_values, color='skyblue')
plt.title("Optimized KNN Performance Metrics", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

## 2. Confusion Matrix
# cm = confusion_matrix(True Labels, Predicted Labels)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
# Use 'Blues' cmap for consistency with the provided example
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False) 
plt.title("Confusion Matrix - Optimized KNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

## 3. ROC Curve
# Calculate False Positive Rate (fpr), True Positive Rate (tpr)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"Optimized KNN (AUC = {roc_auc:.3f})", linewidth=2, color='darkorange')
plt.plot([0,1], [0,1], 'k--', linewidth=1) # Diagonal line for random chance
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Optimized KNN Classifier")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
