"""Telco customer churn prediction: full training pipeline.

Dataset: IBM Telco Customer Churn (7043 customers). Target: Churn (Yes/No).
Cleans the data, encodes categoricals, compares three models with a
held-out test split and 5-fold cross-validation, and saves plots.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load and clean
import os, urllib.request
if not os.path.exists("telco.csv"):
    print("telco.csv not found, downloading ...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "telco.csv")

df = pd.read_csv("telco.csv")
print(f"Loaded {len(df)} customers")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
bad = df["TotalCharges"].isna().sum()
print(f"Rows with empty TotalCharges: {bad} (dropped)")
df = df.dropna(subset=["TotalCharges"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print(f"Churn rate: {df['Churn'].mean():.1%}\n")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

# 2. Compare three models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

results = {}
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_f1": cv_f1,
    }
    fitted[name] = pipe
    r = results[name]
    print(f"{name:20s} acc={r['accuracy']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f} cv_f1={r['cv_f1']:.3f}")

# 3. Plots: confusion matrix of the best-F1 model + metric comparison
best = max(results, key=lambda k: results[k]["f1"])
print(f"\nBest model by F1: {best}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm = confusion_matrix(y_test, fitted[best].predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=["stay", "churn"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"{best} - confusion matrix")

mres = pd.DataFrame(results).T[["accuracy", "f1", "roc_auc", "cv_f1"]]
mres.plot.bar(ax=axes[1])
axes[1].set_ylim(0, 1)
axes[1].set_title("Model comparison")
axes[1].legend(fontsize=8)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("results.svg")
print("Saved results.svg")
