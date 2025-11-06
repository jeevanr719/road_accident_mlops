import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model(data_path="data/processed_accidents.csv", model_path="models/road_accident_model.pkl"):
    df = pd.read_csv(data_path)
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("✅ Model trained successfully!")
    print(classification_report(y_test, y_pred))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
