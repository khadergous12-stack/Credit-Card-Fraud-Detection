# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project that detects fraudulent transactions using advanced classification models and provides an interactive dashboard for real-time prediction and insights.

---

## 🚀 Project Overview

This system predicts whether a credit card transaction is **Fraudulent or Safe** using a trained machine learning model. It also provides a modern dashboard with visual analytics and key insights for better understanding of transaction behavior.

---

## 🧠 Key Features

- 🔍 Real-time fraud prediction
- 📊 Interactive dashboard with charts
- 📈 Fraud probability visualization
- 🧾 Explainable insights (feature impact)
- ⚡ FastAPI backend for prediction API
- 💻 Next.js frontend dashboard
- 🎯 High accuracy model (XGBoost)

---

## 🛠️ Tech Stack

**Machine Learning**
- Python
- Scikit-learn
- XGBoost
- Pandas, NumPy

**Backend**
- FastAPI

**Frontend**
- Next.js
- Chart.js

---

## 📊 Model Details

- Algorithm: **XGBoost Classifier**
- Dataset: Credit Card Fraud Dataset (Kaggle)
- Imbalance handled using: `scale_pos_weight`
- Evaluation Metrics:
  - Accuracy: ~99%
  - AUC Score: ~0.96

---

## 📁 Project Structure


Credit-Card-Fraud-Detection/
│
├── app/
│ └── api.py # FastAPI backend
│
├── src/
│ ├── train.py # Model training
│ ├── predict.py # Prediction logic
│ └── utils.py
│
├── fraud-dashboard/
│ └── app/page.tsx # Frontend dashboard
│
├── models/
│ └── model.pkl
│
├── outputs/
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ └── pr_curve.png
│
└── requirements.txt


---

## ⚙️ How to Run

### 1️⃣ Backend (FastAPI)

```bash
uvicorn app.api:app --reload

👉 Open: http://127.0.0.1:8000/docs

2️⃣ Frontend (Next.js)
cd fraud-dashboard
npm install
npm run dev

👉 Open: http://localhost:3000

📌 Example Inputs
✅ Safe Transaction
Amount: 100
Time: 50000
⚠️ Fraud Transaction
Amount: 15000
Time: 20
⚠️ Note

Dataset is not included due to GitHub size limitations.
You can download it from Kaggle:
👉 https://www.kaggle.com/mlg-ulb/creditcardfraud

🚀 Future Improvements
Live deployment (Vercel + Render)
SHAP-based explainability
Real-time transaction simulation
Alerts & monitoring system
👨‍💻 Author

Khader Gouse
B.Tech (AI/ML)

⭐ If you like this project, give it a star!
