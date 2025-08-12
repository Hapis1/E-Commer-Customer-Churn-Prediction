import pandas as pd
import joblib
import sys

model_path = 'models/churn_model.pkl'
model = joblib.load(model_path)

def predict_churn(input_data):
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data
        
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability

if __name__ == "__main__":
    # Contoh input
    sample_input = {
        "order_count": 3,
        "total_payment": 1500.0,
        "avg_review_score": 4.3,
        "avg_delivery_days": 10.0,
        "days_since_last_purchase": 190
    }

    pred, prob = predict_churn(sample_input)
    label = "Churn" if pred == 1 else "Tidak Churn"
    print(f"Prediksi: {label} (Prob: {prob:.2%})")