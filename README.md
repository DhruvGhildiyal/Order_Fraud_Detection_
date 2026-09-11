# E-Commerce Order Fraud Detection
 
A portfolio project that detects fake orders, return abuse, and multi-account fraud
rings on synthetic e-commerce order data — combining rule-based methods (IQR),
unsupervised machine learning (Isolation Forest), and entity-graph clustering
(shared address/device detection) into a single weighted risk score, deployed as
a real-time FastAPI scoring service with an interactive dashboard.
 
---
 
## Problem
 
E-commerce platforms lose significant revenue to fraud: fake accounts placing
high-value orders, "wardrobing" (buy-and-return abuse), and coordinated fraud
rings using shared addresses or devices across multiple accounts. Manual review
doesn't scale — this project builds an automated risk-scoring pipeline that flags
suspicious orders in real time so only high-risk orders need human review.
 
## Approach
 
Since real labeled fraud data isn't publicly available, this project uses a
10,000-row **synthetic dataset** with 7 deliberately injected, realistic fraud
archetypes (not random noise) at a ~5% fraud rate, matching real-world class
imbalance.
 
The detection pipeline combines four layers:
 
| Layer | Method | Catches |
|---|---|---|
| 1 | **IQR (category-wise)** | Oversized order amounts relative to category norms |
| 2 | **Isolation Forest** | Multivariate anomalies (unsupervised) |
| 3 | **Entity-graph clustering** | Shared device/address/IP across accounts (fraud rings) |
| 4 | **Rule-based flags** | Odd-hour new accounts, return abuse, billing mismatch |
 
All signals are combined into a single **weighted risk score (0–100)**, bucketed
into Low / Medium / High, with a human-readable explanation for each flag.
 
## Results
 
- **ROC-AUC: 0.948** | **PR-AUC: 0.640** (random baseline: 0.049)
- **Precision (fraud): 57%** | **Recall (fraud): 67%** | **F1: 0.616**
- Entity-graph patterns (address/device rings) — **100% recall**, because
  sharing signals are visible even when each individual order looks ordinary
- Known limitation: **account takeover — 3.6% recall.** An old, trusted account
  making a normal-sized order looks fine on every single-row feature; catching
  this needs a per-customer behavioral baseline (a natural next iteration).
## Tech Stack
 
`Python` · `pandas` · `numpy` · `scikit-learn` (Isolation Forest) · `FastAPI` ·
`uvicorn` · `joblib` · Jupyter Notebook
 
## Project Structure
 
```
ecommerce-fraud-detection/
├── data/
│   ├── raw/                    # Original synthetic dataset
│   └── processed/              # Final scored dataset
├── notebooks/
│   └── ecommerce_fraud_detection_code.ipynb   # Full analysis pipeline
├── models/
│   └── fraud_isolation_forest.joblib          # Trained model bundle
├── backend/
│   ├── main.py                 # FastAPI app + dashboard
│   ├── model_utils.py          # Feature engineering functions
│   ├── requirements.txt
│   └── test_request.py         # Sample API test calls
└── README.md
```
 
## How to Run
 
**1. Train / explore the model (optional — pre-trained model is already included)**
```bash
cd notebooks
jupyter notebook ecommerce_fraud_detection_code.ipynb
```
 
**2. Run the backend API + dashboard**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
 
**3. Open the dashboard**
 
Go to `http://127.0.0.1:8000/` in your browser — fill in an order (or use a
Quick Fill example) and click **Score This Order** to see the risk score,
bucket, and explanation live.
 
API docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`.
 
## Key Learnings
 
- **Unsupervised ≠ untestable.** Even without training on labels, holding out
  ground truth for evaluation only (not training) made it possible to measure
  real precision/recall and do honest error analysis.
- **Accuracy is misleading on imbalanced data.** A model that never predicts
  fraud would score 95.1% accuracy here — which is why precision/recall/F1
  are reported instead.
- **Not all fraud looks the same.** Simple statistical outliers (IQR) catch
  obvious cases; entity-graph signals catch coordinated rings; but sophisticated
  fraud (account takeover) needs behavioral, per-customer baselines — a single
  method is never enough.
## Limitations & Next Steps
 
- Trained and evaluated on **synthetic data**; real-world fraud is noisier and
  patterns drift over time, requiring periodic retraining.
- Entity-reuse features (device/address history) require a live database
  lookup in production — the current API accepts these as optional inputs
  with sensible defaults for new customers.
- Next iteration: per-customer behavioral baselines to catch account takeover.
---
 
*This is a portfolio/demo project built to demonstrate an end-to-end fraud
detection pipeline — from data generation through deployment — not a
production-audited system.*
