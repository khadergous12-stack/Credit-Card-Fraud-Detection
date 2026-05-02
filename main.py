from src.preprocess import load_data, split_data
from src.train import train_model

df = load_data("data/creditcard.csv")
X, y = split_data(df)

train_model(X, y)