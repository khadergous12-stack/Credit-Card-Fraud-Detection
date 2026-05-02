import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("🚀 Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

# Reduce size for speed
df = df.sample(n=20000, random_state=42)

X = df.drop("Class", axis=1)
y = df["Class"]

print("🚀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

print("🚀 Training XGBoost model...")
model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 🔥 CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# 🔥 PR CURVE
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("outputs/pr_curve.png")
plt.close()

# 🔥 ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.legend()
plt.title("ROC Curve")
plt.savefig("outputs/roc_curve.png")
plt.close()

# 🔥 SHAP (SAFE VERSION)
print("🚀 Generating SHAP...")

try:
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test[:100])

    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("outputs/shap_summary.png")
    plt.close()

    print("✅ SHAP saved")

except Exception as e:
    print("⚠️ SHAP skipped:", e)

# SAVE MODEL
joblib.dump(model, "models/model.pkl")

print("✅ Model saved")
print("✅ Graphs saved in outputs/")