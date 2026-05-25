import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# load data
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
    ]
]

y = df['Estimated_Profit']

X_train,X_test,y_train,y_test=\
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=RandomForestRegressor()

params={

'n_estimators':[50,100,200],

'max_depth':[5,10,20],

'min_samples_split':[2,5]

}

grid=GridSearchCV(

model,
params,
cv=5,
scoring='r2',
n_jobs=-1

)

grid.fit(
    X_train,
    y_train
)

best=grid.best_estimator_

pred=best.predict(
    X_test
)

print(
    "Best Parameters:"
)

print(
    grid.best_params_
)

print(
    "\nR2:",
    r2_score(
        y_test,
        pred
    )
)

joblib.dump(
    best,
    "models/model.pkl"
)