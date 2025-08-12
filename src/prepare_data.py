import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import os

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/processed/churn_features.csv')

X = df.drop(columns=['churn', 'customer_unique_id'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Data berhasil di-split dan disimpan di data/processed/")