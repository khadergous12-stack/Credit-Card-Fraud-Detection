import pandas as pd
import requests

df = pd.read_csv("data/creditcard.csv")

sample = df.drop("Class", axis=1).iloc[0].tolist()

url = "http://127.0.0.1:8000/predict"

try:
    response = requests.post(url, json={"features": sample})
    print("Prediction:", response.json())
except Exception as e:
    print("Error:", e)