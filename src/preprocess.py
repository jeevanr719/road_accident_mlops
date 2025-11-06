import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data(accidents_path, vehicles_path, casualties_path, nrows=50000):
    accidents = pd.read_csv(accidents_path, nrows=nrows)
    vehicles = pd.read_csv(vehicles_path, nrows=nrows)
    casualties = pd.read_csv(casualties_path, nrows=nrows)
    
    df = pd.merge(vehicles, accidents, on='Accident_Index', how='left')
    df = pd.merge(df, casualties, on='Accident_Index', how='left')
    print(f"Merged shape: {df.shape}")
    return df

def clean_data(df):
    df = df.drop(columns=['Accident_Index'], errors='ignore')
    df = df.dropna(subset=['Accident_Severity'])
    df = df.fillna(df.median(numeric_only=True))
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def save_processed_data(df, output_path="data/processed_accidents.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")

if __name__ == "__main__":
    df = load_data("data/AccidentsBig.csv", "data/VehiclesBig.csv", "data/CasualtiesBig.csv")
    df = clean_data(df)
    save_processed_data(df)
