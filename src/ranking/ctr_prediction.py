import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
# import matplotlib.pyplot as plt

df = pd.read_csv("../../data/ad_10000records.csv")
# print(df.head())


num_cols = ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]
cat_cols = ["Ad Topic Line", "City", "Gender", "Country"]

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["hour"] = df["Timestamp"].dt.hour
df["day"] = df["Timestamp"].dt.dayofweek

num_cols += ["hour", "day"]

X = df[num_cols + cat_cols]
# print(X.head())
y = df["Clicked on Ad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("prep", preproc),
    ("clf", LogisticRegression(max_iter=500))
]).fit(X_train,y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)
