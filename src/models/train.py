import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("data/processed/superstore_processed.csv")

X = df[['Sales', 'Discount', 'Year', 'Month', 'WeekDay']]
y = df['Estimated_Profit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# IMPORTANT FIX
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Profit_Prediction"
)

# 🔥 THIS IS CRITICAL
with mlflow.start_run():

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # log params
    mlflow.log_param("n_estimators", 100)

    # log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    # log model
    mlflow.sklearn.log_model(model, "model")

print("Training complete")