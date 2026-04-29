# Transaction-Fraud-and-Anomaly-Detection-with-RAG-Augmented-LLM-Explainability

A full-stack fraud detection system combining **machine learning**, **deep learning**, **anomaly detection**, **Retrieval-Augmented Generation (RAG)**, and **LLM-based explanations**, wrapped in an interactive **Flask dashboard**.

---

## Project Overview

This system detects fraudulent financial transactions using an ensemble of models and explains predictions using retrieved fraud compliance rules combined with LLM reasoning. It is designed to simulate a real-world fintech fraud monitoring pipeline — from raw transaction ingestion through risk scoring and human-readable explanation.

---

## Key Features

### Machine Learning Models

Three complementary models are combined to maximize detection coverage:

- **Isolation Forest** — Unsupervised anomaly detection that isolates outliers in high-dimensional feature space.
- **XGBoost** — Supervised gradient-boosted classifier trained on labeled fraud data.
- **Autoencoder** — Deep learning model that flags transactions with high reconstruction error as anomalous.

### Ensemble Risk Scoring

Outputs from all three models are combined into a single **final risk score**, categorized as:

| Risk Level | Label |
|---|---|
| 🟢 Low Risk | Normal transaction, no action required |
| 🟠 Medium Risk | Flagged for review |
| 🔴 High Risk | Likely fraudulent, escalate immediately |

### RAG — Retrieval-Augmented Generation

- Fraud compliance rules are stored as structured text documents.
- Semantic search is performed using **SentenceTransformers** embeddings.
- The most relevant rules are retrieved per transaction to ground the explanation.

### LLM Explainability

For every transaction, the system generates a plain-English explanation by combining:

- Model prediction outputs and scores
- Raw transaction features
- Retrieved fraud compliance rules

This produces actionable, auditable reasoning rather than a black-box result.

### Flask Dashboard

An interactive web UI where analysts can:

- Input a transaction index to analyze
- View the ensemble risk score and category
- Inspect individual model contributions
- Read the LLM-generated fraud explanation

---

## System Architecture

```
Raw Data (IEEE-CIS Fraud Dataset)
         ↓
Preprocessing (Cleaning → Encoding → Feature Engineering)
         ↓
ML Models (Isolation Forest + XGBoost + Autoencoder)
         ↓
Ensemble Scoring (Final Risk Score + Category)
         ↓
RAG Retrieval (Relevant Fraud Compliance Rules)
         ↓
LLM Explanation Engine (Human-Readable Output)
         ↓
Flask Dashboard UI
```

---

## Dataset

This project uses the [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection) from Kaggle.

**Download the dataset and place the files as follows:**

```
data/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection-rag.git
cd fraud-detection-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv tf-env
source tf-env/bin/activate        # macOS / Linux
tf-env\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install flask
```

---

## Usage

Run each step in order.

### Step 1 — Preprocess the data

```bash
python src/preprocessing.py
```

### Step 2 — Train the models

```bash
python src/models/isolation_forest.py
python src/models/xgboost_model.py
python src/models/autoencoder.py
python src/models/ensemble.py
```

### Step 3 — Launch the dashboard

```bash
python src/app.py
```

Then open your browser at `http://localhost:5000`.

---

## Project Structure

```
fraud-detection-rag/
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
├── src/
│   ├── preprocessing.py
│   ├── app.py
│   └── models/
│       ├── isolation_forest.py
│       ├── xgboost_model.py
│       ├── autoencoder.py
│       └── ensemble.py
├── requirements.txt
└── README.md
```
