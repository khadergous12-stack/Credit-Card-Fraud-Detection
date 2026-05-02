import pandas as pd
import joblib
import numpy as np

model = joblib.load("models/model.pkl")

# load dataset once
df = pd.read_csv("data/creditcard.csv")

def predict_transaction(features):

    # if user sends empty / fake → use real data row
    if len(features) != 30:
        sample = df.sample(1)
        features = sample.drop("Class", axis=1).values[0]

    X = np.array(features).reshape(1, -1)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    decision = "BLOCK 🚨" if pred == 1 else "ALLOW ✅"

    return int(pred), float(prob), decision