#!/usr/bin/env python3
import argparse, os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(accidents_path, vehicles_path, casualties_path, nrows):
    accidents  = pd.read_csv(accidents_path, nrows=nrows) if accidents_path else None
    vehicles   = pd.read_csv(vehicles_path, nrows=nrows) if vehicles_path else None
    casualties = pd.read_csv(casualties_path, nrows=nrows) if casualties_path else None

    df = vehicles
    if accidents is not None:
        df = pd.merge(df, accidents, on='Accident_Index', how='left')
    if casualties is not None:
        df = pd.merge(df, casualties, on='Accident_Index', how='left')
    print(f"[preprocess] merged shape: {df.shape}")
    return df

def clean_data(df):
    df = df.drop(columns=['Accident_Index'], errors='ignore')
    if 'Accident_Severity' not in df.columns:
        raise RuntimeError("Target column 'Accident_Severity' not found after merge.")
    df = df.dropna(subset=['Accident_Severity'])
    df = df.fillna(df.median(numeric_only=True))
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accidents", default="data/AccidentsBig.csv")
    ap.add_argument("--vehicles", default="data/VehiclesBig.csv")
    ap.add_argument("--casualties", default="data/CasualtiesBig.csv")
    ap.add_argument("--nrows", type=int, default=50000, help="rows per CSV to read (subset for CI/Colab)")
    ap.add_argument("--output", default="data/processed_accidents.csv")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    df = load_data(args.accidents, args.vehicles, args.casualties, args.nrows)
    df = clean_data(df)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[preprocess] saved -> {args.output}")

if __name__ == "__main__":
    main()

