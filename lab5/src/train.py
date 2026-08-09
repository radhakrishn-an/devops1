import json
import joblib
import yaml
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


MODEL_DIR = Path("model")


def main():
    # Read parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    # Create model directory
    MODEL_DIR.mkdir(exist_ok=True)

    # Load training data
    train_df = pd.read_csv("data/train.csv")

    # Separate features and label
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    # Create Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=train_params["n_estimators"],
        max_depth=train_params["max_depth"],
        random_state=train_params["random_state"]
    )

    # Train model
    clf.fit(X_train, y_train)

    # Save trained model
    joblib.dump(
        clf,
        MODEL_DIR / "model.joblib"
    )

    # Save feature names
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(
            list(X_train.columns),
            f
        )

    print(
        "Model trained and saved to "
        "model/model.joblib"
    )


if __name__ == "__main__":
    main()
