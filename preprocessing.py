#Preprocessing

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from google.colab import files

# File path (update if needed)
file_path = "/content/diabetes.csv"

# Load dataset
df = pd.read_csv(file_path)

# Drop duplicates and handle missing values
df = df.drop_duplicates().dropna()

# Encode categorical columns (if any)
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop(columns=df.columns[-1])  # assuming last column is target
y = df[df.columns[-1]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Combine processed data
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['Target'] = y.values

# Save preprocessed dataset
processed_df.to_csv("/content/preprocessed_diabetes_dataset.csv", index=False)

# Download option (optional)
files.download("/content/preprocessed_diabetes_dataset.csv")

# Preview
print("Data Preprocessing Completed!")
processed_df.head()
