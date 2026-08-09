import yaml
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_dataset():
    # Load Boston Housing dataset
    boston = fetch_openml(name="boston", version=1, as_frame=True)

    df = boston.frame.copy()

    # Rename target column to label
    df = df.rename(columns={"MEDV": "target"})

    # Convert regression target into binary classification
    median_price = df["target"].median()
    df["label"] = (df["target"] > median_price).astype(int)

    # Remove original continuous target
    df = df.drop(columns=["target"])

    return df


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    df = load_dataset()

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["label"] = y_train

    test_df = X_test.copy()
    test_df["label"] = y_test

    Path("data").mkdir(exist_ok=True)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Dataset prepared successfully")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Features: {len(X.columns)}")
    print(f"Classes: {sorted(y.unique())}")


if __name__ == "__main__":
    main()
