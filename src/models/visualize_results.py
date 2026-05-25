import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv(
    "data/processed/superstore_processed.csv"
)

X=df[
[
'Sales',
'Discount',
'Year',
'Month',
'WeekDay'
]
]

y=df['Estimated_Profit']

X_train,X_test,y_train,y_test=\
train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

model=joblib.load(
"models/model.pkl"
)

pred=model.predict(X_test)

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
)
)

# Actual vs predicted

plt.figure()

plt.scatter(
y_test,
pred
)

plt.xlabel(
"Actual Profit"
)

plt.ylabel(
"Predicted Profit"
)

plt.title(
"Actual vs Predicted"
)

plt.savefig(
"docs/actual_vs_predicted.png"
)

plt.show()


# Residual plot

residuals=y_test-pred

plt.figure()

plt.scatter(
pred,
residuals
)

plt.axhline(
0
)

plt.xlabel(
"Predicted"
)

plt.ylabel(
"Residual"
)

plt.title(
"Residual Analysis"
)

plt.savefig(
"docs/residual_plot.png"
)

plt.show()


# Feature importance

importance=model.feature_importances_

plt.figure()

plt.bar(
X.columns,
importance
)

plt.xticks(
rotation=45
)

plt.title(
"Feature Importance"
)

plt.savefig(
"docs/feature_importance.png"
)

plt.show()