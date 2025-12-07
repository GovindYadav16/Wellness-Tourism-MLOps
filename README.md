
# 🚀 Wellness Tourism – End-to-End MLOps Pipeline

### **Automated Data → Model → Deployment Pipeline using GitHub Actions, MLflow & Hugging Face**

This repository contains a complete **MLOps workflow** for building, training, tracking, and deploying a **Wellness Tourism Recommendation Model**.
The project automates:

* 📥 Dataset registration to Hugging Face
* 🧹 Data preprocessing
* 🤖 Model training with MLflow tracking
* 📦 Model & artifact upload
* 🌐 Deployment to Hugging Face Space

Everything runs automatically using **GitHub Actions CI/CD**.

---

## 🧠 Project Overview

The goal is to build a model that analyzes **tourism data** and predicts patterns related to wellness travel.
We track the entire pipeline using MLflow and deploy the final model for real-time usage.

---

# 🔧 Tech Stack

| Layer                   | Technologies                 |
| ----------------------- | ---------------------------- |
| **ML Code**             | Python, Pandas, Scikit-Learn |
| **Experiment Tracking** | MLflow                       |
| **Model Registry**      | Hugging Face Hub             |
| **Deployment**          | Hugging Face Space           |
| **CI/CD**               | GitHub Actions               |
| **Orchestration**       | Multi-stage Workflow Jobs    |

---

# 📂 Project Structure

```
Wellness-Tourism-MLOps/
│
├── mlops/
│   ├── data/
│   │   └── tourism.csv              # Dataset
│   │
│   ├── model_building/
│   │   ├── data_register.py         # Upload dataset to HF Hub
│   │   ├── prep.py                  # Data preprocessing
│   │   ├── train.py                 # ML model training & logging
│   │
│   ├── deployment/
│   │   └── hosting.py               # Push app to Hugging Face Space
│   │
│   ├── requirements.txt             # Dependencies
│
├── .github/workflows/
│   └── mlops_pipeline.yml           # CI/CD pipeline automation
│
└── README.md
```

---

# ⚙️ MLOps Pipeline (GitHub Actions)

This project uses a **4-stage CI/CD pipeline**:

---

## **🟦 1. Dataset Registration**

Uploads dataset → Hugging Face Datasets repo

Script:

```
python mlops/model_building/data_register.py
```

---

## **🟩 2. Data Preparation**

Cleans & transforms the dataset.

Script:

```
python mlops/model_building/prep.py
```

---

## **🟧 3. Model Training**

* Trains ML model
* Logs metrics to MLflow
* Saves model artifacts
* Uploads final model to Hugging Face

Script:

```
python mlops/model_building/train.py
```

---

## **🟪 4. Deployment**

Pushes frontend + model to Hugging Face Space.

Script:

```
python mlops/hosting/hosting.py
```

---

# 🔐 GitHub Secrets Required

Go to:

**GitHub Repo → Settings → Secrets & Variables → Actions**

Add:

| Secret Name | Value                                             |
| ----------- | ------------------------------------------------- |
| `HF_TOKEN`  | Your Hugging Face Access Token (write permission) |

---

# ▶️ How to Run Locally

### **1. Clone the repo**

```bash
git clone https://github.com/<your-username>/Wellness-Tourism-MLOps.git
cd Wellness-Tourism-MLOps
```

### **2. Install dependencies**

```bash
pip install -r mlops/requirements.txt
```

### **3. Run the pipeline manually**

```bash
python mlops/model_building/data_register.py
python mlops/model_building/prep.py
python mlops/model_building/train.py
python mlops/hosting/hosting.py
```

---

# 📊 MLflow Tracking

During training:

* Experiments
* Runs
* Parameters
* Metrics (Accuracy, F1, etc.)
* Artifacts (model, plots)

are automatically logged.

---

# 🌐 Deployment

The final application is deployed to Hugging Face Space:

```
https://huggingface.co/spaces/<username>/<space-name>
```

From here, users can interact with the trained model.

---

# 🧪 Model Output

✔ Cleaned dataset
✔ Trained ML model
✔ Evaluation metrics
✔ UI/Hosted demo
✔ Hugging Face model card
✔ Versioned ML artifacts

---

# 🛠 Troubleshooting

### ❌ “dataset repo not found”

Fix:

* Create dataset repo on HF
* Use same username in script
* Ensure your token has **write access**

### ❌ “403 — permission denied”

Fix:

* Regenerate token
* Make sure you saved it as `HF_TOKEN`
* Token must start with `hf_` or `github_pat_` depending on platform

---

# 🙌 Acknowledgements

This project uses:

* 🎯 Hugging Face Hub
* 🎯 MLflow
* 🎯 GitHub Actions
* 🎯 Python ML ecosystem

