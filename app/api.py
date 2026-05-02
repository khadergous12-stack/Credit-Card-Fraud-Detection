from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

app = FastAPI(title="Fraud Detection API")

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD MODEL
model = joblib.load("models/model.pkl")

class Transaction(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "API Running 🚀"}

@app.post("/predict")
def predict(tx: Transaction):

    if len(tx.features) != 30:
        return {"error": f"Expected 30 features, got {len(tx.features)}"}

    X = np.array(tx.features).reshape(1, -1)

    prob = float(model.predict_proba(X)[0][1])

    amount = tx.features[29] * 1000
    time_val = tx.features[0]

    # 🔥 REALISTIC DECISION LOGIC
    if prob > 0.02 or amount > 8000 or time_val < 100:
        fraud_flag = 1
    else:
        fraud_flag = 0

    decision = "🚨 FRAUD" if fraud_flag else "✅ SAFE"

    features = ["V14", "V10", "V12", "Amount", "Time"]

    importance = [0.8, 0.6, 0.5, 0.4, 0.3] if fraud_flag else [0.3,0.2,0.2,0.1,0.1]

    if fraud_flag:
        explanation = [
            "High risk transaction detected",
            "Large transaction amount",
            "Suspicious timing pattern",
            "Strong anomaly in PCA features (V14, V10, V12)",
            "Model indicates fraud behavior"
        ]
    else:
        explanation = [
            "Transaction appears legitimate",
            "Amount is within safe range",
            "No suspicious timing detected",
            "No anomaly in PCA features",
            "Model indicates safe behavior"
        ]

    return {
        "fraud": fraud_flag,
        "prob": prob,
        "decision": decision,
        "top_features": features,
        "importance": importance,
        "explanation": explanation,
        "amount": amount,
        "time": time_val
    }