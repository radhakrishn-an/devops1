import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from pathlib import Path

from src.prepare import load_dataset


def test_load_dataset():
    df = load_dataset()

    # Dataset should contain data
    assert len(df) > 0

    # Dataset must contain the target label
    assert "label" in df.columns

    # Label must contain exactly two classes
    assert df["label"].nunique() == 2

    # Labels must be 0 and 1
    assert set(df["label"].unique()) == {0, 1}


def test_prepare_creates_files():
    # Run the preparation script
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "src/prepare.py"
        ],
        capture_output=True,
        text=True
    )

    # Script should finish successfully
    assert result.returncode == 0

    # Check generated files
    assert Path("data/train.csv").exists()
    assert Path("data/test.csv").exists()


def test_train_and_test_columns_match():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Training and testing datasets should have the same columns
    assert list(train_df.columns) == list(test_df.columns)

    # Both datasets must contain label
    assert "label" in train_df.columns
    assert "label" in test_df.columns
