import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv(
    "data/processed/superstore_processed.csv"
)

X = df[
[
'Sales',
'Discount',
'Year',
'Month',
'WeekDay'
]]

y=df['Estimated_Profit']


X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

model=joblib.load(
"models/model.pkl"
)

pred=model.predict(
X_test
)

print(
"MAE:",
mean_absolute_error(
y_test,
pred
)
)

print(
"R2:",
r2_score(
y_test,
pred
))