import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ===========================================================
# 💡 Ranking metrics
# ===========================================================

def precision_at_k(y_true, y_scores, k):
    idx = np.argsort(y_scores)[::-1][:k]
    return y_true.iloc[idx].mean()

def recall_at_k(y_true, y_scores, k):
    idx = np.argsort(y_scores)[::-1][:k]
    return y_true.iloc[idx].sum() / y_true.sum()

def lift_at_k(y_true, y_scores, k):
    base = y_true.mean()
    return precision_at_k(y_true, y_scores, k) / base


# ===========================================================
# 📊 Evaluation function
# ===========================================================

def evaluate_ctr_model(model_name, y_true, y_pred, y_scores, k=500):
    print("=" * 70)
    print(f"📌 MODEL EVALUATION: {model_name}")
    print("=" * 70)

    auc = roc_auc_score(y_true, y_scores)
    ll  = log_loss(y_true, y_scores)

    print(f"AUC:          {auc:.5f}")
    print(f"Log Loss:     {ll:.5f}")
    print(f"Baseline CTR: {y_true.mean():.5f}")

    print("\nTop-K Ranking Metrics:")
    print(f" Precision@{k}:  {precision_at_k(y_true, y_scores, k):.5f}")
    print(f" Recall@{k}:     {recall_at_k(y_true, y_scores, k):.5f}")
    print(f" Lift@{k}:       {lift_at_k(y_true, y_scores, k):.2f}x")

    print("\nConfusion Matrix (threshold = 0.5):")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("=" * 70 + "\n")
    return auc, ll


# ===========================================================
# 📂 Load Data
# ===========================================================

cols = (["click"] +
        [f"I{i}" for i in range(1,14)] +
        [f"C{i}" for i in range(1,27)])

df = pd.read_csv("../../data/day_0_100k", sep="\t", names=cols, header=None)

num_cols = [f"I{i}" for i in range(1,14)]
cat_cols = [f"C{i}" for i in range(1,27)]
y = df["click"]

# ===========================================================
# 1️⃣ Logistic Regression — NUMERICAL ONLY
# ===========================================================

X_num = df[num_cols]
X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)

preproc_num = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols)
])

model_lr_num = Pipeline([
    ("prep", preproc_num),
    ("clf", LogisticRegression(max_iter=1000))
]).fit(X_train, y_train)

y_pred = model_lr_num.predict(X_test)
y_proba = model_lr_num.predict_proba(X_test)[:, 1]

evaluate_ctr_model("LR Numerical Only", y_test, y_pred, y_proba)


# ===========================================================
# 2️⃣ Logistic Regression — LABEL ENCODED CATEGORICALS
# ===========================================================

X_full = df[num_cols + cat_cols]
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# Label encoding
encoders = {}
for col in cat_cols:
    le = LabelEncoder()

    # Fit only on training data
    le.fit(X_train[col].astype(str))

    # Add UNK token
    classes = list(le.classes_)
    if "UNK" not in classes:
        classes.append("UNK")
    le.classes_ = np.array(classes)

    # Replace unseen in test
    X_test[col] = X_test[col].astype(str).where(
        X_test[col].astype(str).isin(le.classes_), "UNK"
    )

    # Replace unseen in train
    X_train[col] = X_train[col].astype(str).where(
        X_train[col].astype(str).isin(le.classes_), "UNK"
    )

    # Transform
    X_train[col] = le.transform(X_train[col])
    X_test[col]  = le.transform(X_test[col])

    encoders[col] = le

preproc_cat = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", "passthrough", cat_cols)
])

model_lr_cat = Pipeline([
    ("prep", preproc_cat),
    ("clf", LogisticRegression(max_iter=1000))
]).fit(X_train, y_train)

y_pred = model_lr_cat.predict(X_test)
y_proba = model_lr_cat.predict_proba(X_test)[:, 1]

evaluate_ctr_model("LR with Categories (Label Encoded)", y_test, y_pred, y_proba)


# ===========================================================
# 3️⃣ Logistic Regression — TOP-K + ONE-HOT
# ===========================================================

def top_k_transform(df, cols, frac=0.01):
    df = df.copy()
    topk = {}
    for col in cols:
        df[col] = df[col].astype(str)
        freq = df[col].value_counts()
        k = max(1, int(len(freq) * frac))
        keep = set(freq.index[:k])
        df[col] = df[col].where(df[col].isin(keep), "__OTHER__")
        topk[col] = keep
    return df, topk

def apply_top_k(df, cols, topk):
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].where(df[col].isin(topk[col]), "__OTHER__")
    return df

# top-k version
X_train_k, topk = top_k_transform(X_train, cat_cols, frac=0.01)
X_test_k = apply_top_k(X_test, cat_cols, topk)

preproc_topk = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model_topk = Pipeline([
    ("prep", preproc_topk),
    ("clf", LogisticRegression(max_iter=1000))
]).fit(X_train_k, y_train)

y_pred = model_topk.predict(X_test_k)
y_proba = model_topk.predict_proba(X_test_k)[:, 1]

evaluate_ctr_model("LR Top-K Encoding", y_test, y_pred, y_proba)


# ===========================================================
# 4️⃣ XGBoost — ALL FEATURES LABEL ENCODED (Correct!)
# ===========================================================

X_xgb = df[num_cols + cat_cols].copy()
y_xgb = df["click"]

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42
)

# Label encode for XGB
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()

    X_train_xgb[col] = le.fit_transform(X_train_xgb[col].astype(str))

    classes = list(le.classes_)
    if "UNK" not in classes:
        classes.append("UNK")
    le.classes_ = np.array(classes)

    X_test_xgb[col] = X_test_xgb[col].astype(str).where(
        X_test_xgb[col].isin(le.classes_), "UNK"
    )

    X_test_xgb[col] = le.transform(X_test_xgb[col])
    label_encoders[col] = le


# -----------------
# XGB Baseline
# -----------------
model_xgb = XGBClassifier(
    tree_method="hist",
    eval_metric="logloss"
).fit(X_train_xgb, y_train_xgb)

y_pred = model_xgb.predict(X_test_xgb)
y_proba = model_xgb.predict_proba(X_test_xgb)[:,1]

evaluate_ctr_model("XGBoost Baseline (Label Encoded)", y_test_xgb, y_pred, y_proba)


# -----------------
# XGB No scaling (same data)
# -----------------
model_xgb2 = XGBClassifier(
    tree_method="hist",
    eval_metric="logloss"
).fit(X_train_xgb, y_train_xgb)

y_pred = model_xgb2.predict(X_test_xgb)
y_proba = model_xgb2.predict_proba(X_test_xgb)[:,1]

evaluate_ctr_model("XGBoost No Scaling", y_test_xgb, y_pred, y_proba)


# -----------------
# XGB Tuned
# -----------------
model_xgb3 = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1.0,
    reg_lambda=2.0,
    tree_method="hist",
    max_bin=256,
    eval_metric="logloss"
).fit(X_train_xgb, y_train_xgb)

y_pred = model_xgb3.predict(X_test_xgb)
y_proba = model_xgb3.predict_proba(X_test_xgb)[:,1]

evaluate_ctr_model("XGBoost Tuned", y_test_xgb, y_pred, y_proba)
