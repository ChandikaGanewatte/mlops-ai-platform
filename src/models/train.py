import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load processed data
df = pd.read_csv(
    "data/processed/superstore_processed.csv"
)

# Features
X = df[
    [
        'Sales',
        'Discount',
        'Year',
        'Month',
        'WeekDay'
    ]
]

# Target
y = df['Estimated_Profit']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(
    X_train,
    y_train
)

# Save model
joblib.dump(
    model,
    "models/model.pkl"
)

print("Model Training Complete")