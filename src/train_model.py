import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

os.makedirs('models', exist_ok=True)

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100,200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Evaluasi
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameter:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(best_model, "models/churn_model.pkl")
print("Berhasil disimpan di models/")