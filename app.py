import streamlit as st
import pandas as pd
import joblib
from src.predict import predict_churn

model = joblib.load("models/churn_model.pkl")

st.set_page_config(page_title="E-Commerce Churn Prediction", page_icon="📊", layout="centered")
st.title("E-Commerce Customer Churn Prediction")

menu = st.sidebar.selectbox("Menu", ["Prediksi Manual", "Prediksi dari CSV"])

if menu == "Prediksi Manual":
    st.subheader("Input Data Pelanggan")

    order_count = st.number_input("Jumlah Order", min_value=0, value=3)
    total_payment = st.number_input("Total Pembayaran", min_value=0.0, value=1500.0)
    avg_review_score = st.number_input("Rata-rata Skor Review", min_value=0.0, max_value=5.0, value=4.3)
    avg_delivery_days = st.number_input("Rata-rata Hari Pengiriman", min_value=0.0, value=10.0)
    days_since_last_purchase = st.number_input("Hari sejak pembelian terakhir", min_value=0, value=190)

    if st.button("Prediksi Churn"):
        input_data = {
            "order_count": order_count,
            "total_payment": total_payment,
            "avg_review_score": avg_review_score,
            "avg_delivery_days": avg_delivery_days,
            "days_since_last_purchase": days_since_last_purchase
        }
        pred, prob = predict_churn(input_data)
        label = "🔥 Churn" if pred == 1 else "✅ Tidak Churn"
        st.write(f"**Prediksi:** {label}")
        st.write(f"**Probabilitas Churn:** {prob:.2%}")

elif menu == "Prediksi dari CSV":
    st.subheader("Upload File CSV")
    
    example_data = pd.read_csv('data/processed/churn_features.csv')
    csv_data = example_data.to_csv(index=False).encode("utf-8")
    
    st.download_button(
        label="Download Contoh CSV",
        data=csv_data,
        file_name="contoh_dataset_churn.csv",
        mime="text/csv"
    )
    st.caption("Gunakan file contoh ini untuk mencoba memprediksi.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Diupload:")
        st.dataframe(df.head())

        if st.button("Prediksi CSV"):
            preds = model.predict(df)
            probs = model.predict_proba(df)[:, 1]
            df["prediksi_churn"] = preds
            df["prob_churn"] = probs
            st.write("📊 Hasil Prediksi:")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Hasil", csv, "hasil_prediksi.csv", "text/csv")