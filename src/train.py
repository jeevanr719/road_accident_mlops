#!/usr/bin/env python3
import argparse, os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_accidents.csv")
    ap.add_argument("--model_path", default="models/road_accident_model.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)
    report = classification_report(yte, ypred, output_dict=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(model, args.model_path)
    with open("models/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[train] model saved:", args.model_path)
    print("[train] f1-weighted:", round(report["weighted avg"]["f1-score"], 4))

if __name__ == "__main__":
    main()

