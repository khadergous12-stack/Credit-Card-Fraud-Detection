import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def split_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y