# Data manipulation
import pandas as pd
import os

# Train/Test split
from sklearn.model_selection import train_test_split

# Hugging Face authentication
from huggingface_hub import HfApi

# HF API with token from GitHub Actions secret
api = HfApi(token=os.getenv("HF_TOKEN"))

# Path to dataset stored in Hugging Face Datasets
DATASET_PATH = "hf://datasets/yadavgovind/wellness-tourism-space/tourism.csv"

# Load dataset
tourism_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# -----------------------------
# Target variable
# -----------------------------
target = "ProdTaken"   # 1 = Product Taken, 0 = Not Taken

# Identify numerical & categorical features automatically
numeric_features = tourism_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = tourism_df.select_dtypes(include=['object']).columns.tolist()

# Remove unwanted ID columns if present
drop_cols = ["CustomerID", "Unnamed: 0"]
numeric_features = [col for col in numeric_features if col not in drop_cols]

# Define feature matrix (X) and target (y)
X = tourism_df[numeric_features + categorical_features]
y = tourism_df[target]

# -----------------------------
# Train-Test Split
# -----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save files locally for upload
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# -----------------------------
# Upload files to Hugging Face
# -----------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="yadavgovind/wellness-tourism-space",
        repo_type="dataset",
    )

print("Dataset split & uploaded to Hugging Face successfully.")
