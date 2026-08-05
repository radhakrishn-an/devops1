"""
Stage 4 : Model Building
------------------------
Train Random Forest Regressor
"""

import yaml
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


def load_params(path="params.yaml"):

    with open(path) as f:

        return yaml.safe_load(f)


def load_train_data():

    return pd.read_csv(
        "data/features/train.csv"
    )


def train_model(df, n_estimators, max_depth, random_state):

    X = df.drop(columns=["target"])

    y = df["target"]

    model = RandomForestRegressor(

        n_estimators=n_estimators,

        max_depth=max_depth,

        random_state=random_state

    )

    model.fit(X, y)

    return model


def save_model(model):

    joblib.dump(model, "model.pkl")

    print("Model Saved Successfully.")


def main():

    params = load_params()["model_building"]

    df = load_train_data()

    model = train_model(
        df,
        params["n_estimators"],
        params["max_depth"],
        params["random_state"]
    )

    save_model(model)


if __name__ == "__main__":
    main()
