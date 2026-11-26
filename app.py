"""Streamlit app: telco customer churn prediction.

Trains a logistic regression pipeline on the IBM Telco churn dataset
(included as telco.csv) and estimates churn probability for a customer
you describe in the sidebar.

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@st.cache_resource
def train_model():
    import os, urllib.request
    if not os.path.exists("telco.csv"):
        print("telco.csv not found, downloading ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
            "telco.csv")

    df = pd.read_csv("telco.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()
    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num)])),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    stats = {"accuracy": accuracy_score(y_te, pred),
             "f1": f1_score(y_te, pred),
             "roc_auc": roc_auc_score(y_te, proba)}
    return pipe, df, stats


pipe, df, stats = train_model()

st.title("Telco customer churn prediction")
st.write("Practice project on the IBM Telco churn dataset "
         f"({len(df)} customers, churn rate {df['Churn'].mean():.0%}).")
st.write(f"Logistic regression on a held-out test split - accuracy "
         f"**{stats['accuracy']:.3f}**, F1 **{stats['f1']:.3f}**, "
         f"ROC AUC **{stats['roc_auc']:.3f}**")

st.sidebar.header("Describe the customer")
modes = df.drop(columns=["customerID", "Churn"]).mode().iloc[0]
row = dict(modes)
row["tenure"] = st.sidebar.slider("Tenure (months)", 0, 72, 12)
row["MonthlyCharges"] = st.sidebar.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
row["TotalCharges"] = float(row["tenure"]) * row["MonthlyCharges"]
row["Contract"] = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
row["InternetService"] = st.sidebar.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
row["PaymentMethod"] = st.sidebar.selectbox("Payment method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
row["Partner"] = st.sidebar.selectbox("Has a partner", ["Yes", "No"])
row["Dependents"] = st.sidebar.selectbox("Has dependents", ["No", "Yes"])
row["SeniorCitizen"] = st.sidebar.selectbox("Senior citizen", [0, 1])
row["PaperlessBilling"] = st.sidebar.selectbox("Paperless billing", ["Yes", "No"])

p = pipe.predict_proba(pd.DataFrame([row]))[0][1]
st.subheader("Prediction")
if p >= 0.5:
    st.error(f"Likely to CHURN (probability {p:.1%})")
else:
    st.success(f"Likely to stay (churn probability {p:.1%})")
st.caption("Educational model trained on a 2020 sample dataset - not a real business tool.")
