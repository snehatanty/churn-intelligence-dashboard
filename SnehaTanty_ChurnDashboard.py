"""
Telecom Customer Churn Intelligence Dashboard
---------------------------------------------
Business-intelligence project: Data -> Insight -> Decision -> Action.

Backend  : pandas + scikit-learn (churn prediction model)
Frontend : Streamlit + Plotly (3-page decision dashboard)

Run:  streamlit run churn_intelligence_dashboard.py
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOCAL_FILE = "Telco-Customer-Churn.csv"
FALLBACK_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

st.set_page_config(page_title="Churn Intelligence Dashboard", page_icon="📉", layout="wide")


# ----------------------------------------------------------------------------
# 1. DATA LAYER
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw(uploaded_bytes=None):
    """Load the dataset: upload -> local file -> public mirror of the same data."""
    if uploaded_bytes is not None:
        import io

        return pd.read_csv(io.BytesIO(uploaded_bytes))
    if os.path.exists(LOCAL_FILE):
        return pd.read_csv(LOCAL_FILE)
    return pd.read_csv(FALLBACK_URL)


@st.cache_data(show_spinner=False)
def clean(df_raw):
    """Clean the raw data and create business-friendly fields."""
    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
    )
    return df


# ----------------------------------------------------------------------------
# 2. MODEL LAYER
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training churn model...")
def train_model(df):
    features = df.drop(columns=["customerID", "Churn", "ChurnFlag", "TenureGroup"])
    target = df["ChurnFlag"]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in features.columns if c not in num_cols]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    model = Pipeline(
        [("prep", pre), ("clf", GradientBoostingClassifier(random_state=42))]
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
        "F1 (churn)": f1_score(y_te, pred),
        "cm": confusion_matrix(y_te, pred),
    }
    scored = df.copy()
    scored["ChurnProbability"] = model.predict_proba(features)[:, 1]

    names = model.named_steps["prep"].get_feature_names_out()
    imp = (
        pd.DataFrame(
            {
                "Feature": [n.split("__", 1)[1] for n in names],
                "Importance": model.named_steps["clf"].feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .head(10)
    )
    return model, metrics, scored, imp, features.columns.tolist()


def churn_rate_by(df, col):
    out = (
        df.groupby(col, observed=True)
        .agg(Customers=("ChurnFlag", "size"), ChurnRate=("ChurnFlag", "mean"))
        .reset_index()
    )
    out["ChurnRate"] = out["ChurnRate"] * 100
    return out


# ----------------------------------------------------------------------------
# 3. LOAD EVERYTHING
# ----------------------------------------------------------------------------
st.sidebar.title("📉 Churn Intelligence")
upload = st.sidebar.file_uploader("Optional: upload your own CSV (same columns)", type="csv")
try:
    raw = load_raw(upload.getvalue() if upload else None)
except Exception as exc:  # network or file problem
    st.error(f"Could not load data: {exc}")
    st.stop()

data = clean(raw)
model, metrics, scored, importance, feature_cols = train_model(data)

page = st.sidebar.radio(
    "Dashboard page",
    ["1. Executive Overview", "2. Churn Drivers & Segments", "3. Risk, Opportunity & Action"],
)
st.sidebar.markdown("---")
contract_filter = st.sidebar.multiselect(
    "Filter by contract", sorted(data["Contract"].unique()), default=sorted(data["Contract"].unique())
)
view = scored[scored["Contract"].isin(contract_filter)]
if view.empty:
    st.warning("Select at least one contract type.")
    st.stop()

# Core KPIs (computed on the filtered view)
total_customers = len(view)
churn_rate = view["ChurnFlag"].mean() * 100
mrr = view["MonthlyCharges"].sum()
mrr_lost = view.loc[view["ChurnFlag"] == 1, "MonthlyCharges"].sum()
avg_tenure = view["tenure"].mean()
arpu = view["MonthlyCharges"].mean()

# ----------------------------------------------------------------------------
# PAGE 1 - EXECUTIVE OVERVIEW
# ----------------------------------------------------------------------------
if page.startswith("1"):
    st.title("Executive Overview")
    st.caption("What is happening to our customer base?")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Churn rate", f"{churn_rate:.1f}%")
    c3.metric("Monthly revenue", f"${mrr:,.0f}")
    c4.metric("Monthly revenue lost to churn", f"${mrr_lost:,.0f}")
    c5.metric("Avg. revenue / customer", f"${arpu:,.2f}")

    left, right = st.columns(2)
    with left:
        by_ten = churn_rate_by(view, "TenureGroup")
        fig = px.bar(
            by_ten, x="TenureGroup", y="ChurnRate", text=by_ten["ChurnRate"].round(1),
            title="Churn rate by customer tenure (%)", color="ChurnRate",
            color_continuous_scale="Reds",
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Tenure", yaxis_title="Churn %")
        st.plotly_chart(fig)
    with right:
        by_con = churn_rate_by(view, "Contract")
        fig = px.bar(
            by_con, x="Contract", y="ChurnRate", text=by_con["ChurnRate"].round(1),
            title="Churn rate by contract type (%)", color="ChurnRate",
            color_continuous_scale="Reds",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn %")
        st.plotly_chart(fig)

    trend = view.groupby("tenure").agg(Churn=("ChurnFlag", "mean")).reset_index()
    trend["Churn"] *= 100
    trend["Smoothed"] = trend["Churn"].rolling(6, min_periods=1).mean()
    fig = px.line(trend, x="tenure", y="Smoothed", title="Churn trend across customer lifetime (6-month rolling avg)")
    fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Churn %")
    st.plotly_chart(fig)

    st.info(
        f"**Headline:** {churn_rate:.1f}% of customers have churned, taking "
        f"**${mrr_lost:,.0f}/month** of recurring revenue with them. "
        "Churn is concentrated early in the customer lifetime."
    )

# ----------------------------------------------------------------------------
# PAGE 2 - DRIVERS & SEGMENTS
# ----------------------------------------------------------------------------
elif page.startswith("2"):
    st.title("Churn Drivers & Segments")
    st.caption("Why are customers leaving?")

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
            title="Top 10 drivers of churn (model feature importance)",
        )
        st.plotly_chart(fig)
    with right:
        dim = st.selectbox(
            "Explore churn rate by:",
            ["InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
             "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents", "StreamingTV"],
        )
        seg = churn_rate_by(view, dim)
        fig = px.bar(seg, x=dim, y="ChurnRate", text=seg["ChurnRate"].round(1),
                     title=f"Churn rate by {dim} (%)", color="ChurnRate", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn %")
        st.plotly_chart(fig)

    fig = px.box(
        view, x="Churn", y="MonthlyCharges", color="Churn",
        title="Monthly charges: churned vs retained customers",
    )
    st.plotly_chart(fig)

    heat = view.pivot_table(index="Contract", columns="InternetService", values="ChurnFlag", aggfunc="mean") * 100
    fig = px.imshow(heat.round(1), text_auto=True, color_continuous_scale="Reds",
                    title="Churn % - Contract x Internet service")
    st.plotly_chart(fig)

    st.subheader("Model performance (20% hold-out test set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    m2.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
    m3.metric("F1 (churn class)", f"{metrics['F1 (churn)']:.3f}")

# ----------------------------------------------------------------------------
# PAGE 3 - RISK, OPPORTUNITY, ACTION
# ----------------------------------------------------------------------------
else:
    st.title("Risk, Opportunity & Action")
    st.caption("What should management do next?")

    active = view[view["ChurnFlag"] == 0].copy()
    threshold = st.slider("High-risk threshold (churn probability)", 0.3, 0.9, 0.5, 0.05)
    high_value_cut = active["MonthlyCharges"].median()
    at_risk = active[active["ChurnProbability"] >= threshold]
    hv_at_risk = at_risk[at_risk["MonthlyCharges"] >= high_value_cut]

    r1, r2, r3 = st.columns(3)
    r1.metric("Active customers at high risk", f"{len(at_risk):,}")
    r2.metric("High-value & at risk", f"{len(hv_at_risk):,}")
    r3.metric("Monthly revenue at risk", f"${at_risk['MonthlyCharges'].sum():,.0f}")

    fig = px.histogram(
        active, x="ChurnProbability", nbins=30,
        title="Predicted churn probability - currently active customers",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

    st.subheader("Top 15 customers to contact first")
    st.caption("Active customers ranked by churn probability x monthly revenue.")
    top = hv_at_risk.assign(Priority=lambda d: d["ChurnProbability"] * d["MonthlyCharges"]).sort_values(
        "Priority", ascending=False
    ).head(15)
    st.dataframe(
        top[["customerID", "Contract", "InternetService", "tenure", "MonthlyCharges", "ChurnProbability"]]
        .style.format({"ChurnProbability": "{:.0%}", "MonthlyCharges": "${:.2f}"}),
        
    )

    st.subheader("Fact -> Insight -> Opportunity -> Action")
    mtm = view[view["Contract"] == "Month-to-month"]
    longc = view[view["Contract"] != "Month-to-month"]
    ec = view[view["PaymentMethod"] == "Electronic check"]
    new = view[view["tenure"] <= 12]
    st.markdown(
        f"""
| Step | Finding |
|---|---|
| **Risk** | Month-to-month customers churn at **{mtm['ChurnFlag'].mean()*100:.1f}%** vs **{longc['ChurnFlag'].mean()*100:.1f}%** on 1-2 year contracts. Customers in their first year churn at **{new['ChurnFlag'].mean()*100:.1f}%**. |
| **Opportunity** | Electronic-check payers churn at **{ec['ChurnFlag'].mean()*100:.1f}%**; moving them to auto-pay and adding tech support / online security bundles targets a high-leakage segment. |
| **Action 1** | Offer a discount for switching month-to-month customers to a 1-year contract. |
| **Action 2** | Launch an onboarding and check-in programme for customers in their first 12 months. |
| **Action 3** | Run a retention call/offer for the {len(hv_at_risk):,} high-value at-risk customers listed above. |
"""
    )

    with st.expander("Score a single customer (what-if tool)"):
        a, b, c = st.columns(3)
        contract = a.selectbox("Contract", sorted(data["Contract"].unique()))
        internet = b.selectbox("Internet service", sorted(data["InternetService"].unique()))
        payment = c.selectbox("Payment method", sorted(data["PaymentMethod"].unique()))
        tenure = a.slider("Tenure (months)", 0, 72, 6)
        monthly = b.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
        tech = c.selectbox("Tech support", sorted(data["TechSupport"].unique()))
        base = data[feature_cols].mode().iloc[0].to_dict()  # most common value for other fields
        base.update(
            Contract=contract, InternetService=internet, PaymentMethod=payment, tenure=tenure,
            MonthlyCharges=monthly, TotalCharges=monthly * max(tenure, 1), TechSupport=tech,
        )
        prob = model.predict_proba(pd.DataFrame([base])[feature_cols])[0, 1]
        st.metric("Predicted churn probability", f"{prob:.0%}")
