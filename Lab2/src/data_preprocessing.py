"""
Stage 2: Data Preprocessing
---------------------------
Reads the raw Boston Housing dataset,
cleans column names,
handles duplicates and missing values,
and saves the processed data.

Input:
    data/raw/data.csv

Output:
    data/processed/data.csv
"""

import os
import pandas as pd


def load_raw_data(path="data/raw/data.csv"):

    df = pd.read_csv(path)

    print(f"[data_preprocessing] Loaded raw data (shape={df.shape})")

    return df


def clean_data(df):

    df.columns = [
        col.strip().replace(" ", "_").lower()
        for col in df.columns
    ]

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    if before != after:

        print(f"Removed {before-after} duplicate rows")

    if df.isnull().sum().sum() > 0:

        numeric_cols = df.select_dtypes(include="number").columns

        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].median()
        )

        print("Missing values filled.")

    df["target"] = df["target"].astype(float)

    return df


def save_processed_data(df, out_dir="data/processed"):

    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, "data.csv")

    df.to_csv(output_path, index=False)

    print(f"[data_preprocessing] Saved -> {output_path}")


def main():

    df = load_raw_data()

    df = clean_data(df)

    save_processed_data(df)


if __name__ == "__main__":
    main()
