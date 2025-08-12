# E-Commerce Customer Churn Prediction

Proyek ini adalah implementasi **Machine Learning** untuk memprediksi kemungkinan pelanggan berhenti (churn) dari layanan e-commerce. Model ini menggunakan data historis perilaku pelanggan untuk mengidentifikasi pola yang berkaitan dengan churn.

## 📌 Fitur Utama
- **Data Preprocessing**: Pembersihan data, penanganan missing values, encoding variabel kategori, dan normalisasi.
- **Exploratory Data Analysis (EDA)**: Analisis distribusi, korelasi antar fitur, dan visualisasi data.
- **Modeling**: Menggunakan algoritma Machine Learning seperti Logistic Regression, Random Forest, dan XGBoost.
- **Evaluasi Model**: Menggunakan metrik seperti Accuracy, Precision, Recall, F1-Score, dan ROC-AUC.
- **Prediksi Baru**: Dapat memprediksi churn pelanggan baru berdasarkan input data.

## 🛠️ Teknologi yang Digunakan
- **Python 3.x**
- **Pandas** & **NumPy** → Manipulasi dan analisis data
- **Matplotlib** & **Seaborn** → Visualisasi data
- **Scikit-learn** → Machine Learning

## 📂 Struktur Proyek
.
├── data/ # Dataset mentah dan hasil preprocessing
├── notebooks/ # Notebook Jupyter untuk eksplorasi dan modeling
├── models/ # Model yang telah dilatih (pickle/joblib)
├── src/ # Script Python untuk training, preprocessing, dan prediksi
├── README.md # Dokumentasi proyek
└── requirements.txt # Daftar dependencies

## 📊 Alur Pengerjaan
1. **Pengumpulan Data** → Menggunakan dataset churn pelanggan dari e-commerce.
2. **Data Cleaning** → Menghapus duplikasi, mengisi missing values, encoding kategori.
3. **EDA** → Analisis pola dan distribusi data.
4. **Feature Engineering** → Membuat fitur baru dan memilih fitur terbaik.
5. **Model Training** → Melatih model dengan beberapa algoritma.
6. **Evaluasi Model** → Memilih model dengan performa terbaik.
7. **Deployment (Opsional)** → Menyediakan API untuk prediksi churn.

## 🚀 Cara Menjalankan Proyek
1. Clone repository ini:
   ```bash
   git clone https://github.com/username/repo-name.git
Masuk ke direktori proyek:
cd repo-name

Install dependencies:
pip install -r requirements.txt

Jalankan notebook atau script untuk training:
jupyter notebook
atau
python src/train_model.py
📜 Lisensi
Proyek ini menggunakan lisensi MIT — silakan gunakan dan modifikasi sesuai kebutuhan.

