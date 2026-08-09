import sys
import json
import yaml
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def main():

    # Read evaluation parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    evaluate_params = params["evaluate"]

    # Load trained model
    clf = joblib.load("model/model.joblib")

    # Load test dataset
    test_df = pd.read_csv("data/test.csv")

    # Separate features and label
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # Make predictions
    preds = clf.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Display metrics
    print(json.dumps(metrics, indent=2))

    # Quality gate
    if metrics["accuracy"] < evaluate_params["min_accuracy"]:

        print(
            f"FAIL: accuracy "
            f"{metrics['accuracy']:.4f} "
            f"is below gate "
            f"{evaluate_params['min_accuracy']}"
        )

        sys.exit(1)

    print("PASS: model cleared the quality gate")


if __name__ == "__main__":
    main()
