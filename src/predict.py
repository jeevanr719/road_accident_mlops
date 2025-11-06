import pandas as pd
import joblib

def predict(sample_data, model_path="models/road_accident_model.pkl"):
    model = joblib.load(model_path)
    prediction = model.predict(sample_data)
    return prediction

if __name__ == "__main__":
    sample = pd.read_csv("data/processed_accidents.csv").iloc[[0]].drop("Accident_Severity", axis=1)
    print("Predicted Severity:", predict(sample)[0])
