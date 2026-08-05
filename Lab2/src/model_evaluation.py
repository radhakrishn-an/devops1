"""
Stage 5 : Model Evaluation
--------------------------
Evaluates the trained regression model.
"""

import json
import joblib
import pandas as pd

from math import sqrt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def load_model():

    return joblib.load("model.pkl")


def load_test_data():

    return pd.read_csv(
        "data/features/test.csv"
    )


def evaluate(model, df):

    X = df.drop(columns=["target"])

    y = df["target"]

    pred = model.predict(X)

    mse = mean_squared_error(y, pred)

    metrics = {

        "mae": mean_absolute_error(y, pred),

        "mse": mse,

        "rmse": sqrt(mse),

        "r2_score": r2_score(y, pred)

    }

    return metrics


def save_metrics(metrics):

    with open("metrics.json", "w") as f:

        json.dump(metrics, f, indent=4)

    print(metrics)


def main():

    model = load_model()

    df = load_test_data()

    metrics = evaluate(model, df)

    save_metrics(metrics)


if __name__ == "__main__":
    main()
