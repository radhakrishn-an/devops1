"""
Stage 3: Feature Engineering
----------------------------
Reads the processed dataset,
splits into train and test,
scales features,
and saves them.

Output:
    data/features/train.csv
    data/features/test.csv
    data/features/scaler.pkl
"""

import os
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params(path="params.yaml"):

    with open(path) as f:

        return yaml.safe_load(f)


def load_processed_data(path="data/processed/data.csv"):

    return pd.read_csv(path)


def build_features(df, test_size, random_state):

    X = df.drop(columns=["target"])

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns
    )

    train = X_train.copy()

    train["target"] = y_train.reset_index(drop=True)

    test = X_test.copy()

    test["target"] = y_test.reset_index(drop=True)

    return train, test, scaler


def save_features(train, test, scaler):

    os.makedirs("data/features", exist_ok=True)

    train.to_csv("data/features/train.csv", index=False)

    test.to_csv("data/features/test.csv", index=False)

    joblib.dump(
        scaler,
        "data/features/scaler.pkl"
    )

    print("Feature Engineering Completed.")


def main():

    params = load_params()["feature_engineering"]

    df = load_processed_data()

    train, test, scaler = build_features(
        df,
        params["test_size"],
        params["random_state"]
    )

    save_features(train, test, scaler)


if __name__ == "__main__":
    main()
