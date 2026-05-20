import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="MLOPS BI Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/superstore_processed.csv")
    return df

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 MLOps Control Panel")

page = st.sidebar.radio(
    "Navigation",
    ["📈 Dashboard Overview", "🔮 Profit Prediction"]
)

# -----------------------------
# PAGE 1 - DASHBOARD
# -----------------------------
if page == "📈 Dashboard Overview":

    st.title("📊 Business Intelligence Dashboard")

    # KPI METRICS
    total_sales = df["Sales"].sum()
    avg_sales = df["Sales"].mean()
    total_profit = df["Estimated_Profit"].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{total_sales:,.2f}")
    col2.metric("Average Sales", f"{avg_sales:,.2f}")
    col3.metric("Estimated Profit", f"{total_profit:,.2f}")

    st.divider()

    # -----------------------------
    # SALES BY CATEGORY
    # -----------------------------
    st.subheader("Sales by Category")

    fig, ax = plt.subplots()
    df.groupby("Category")["Sales"].sum().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # MONTHLY TREND
    # -----------------------------
    st.subheader("Monthly Sales Trend")

    monthly = df.groupby("Month")["Sales"].sum()

    fig2, ax2 = plt.subplots()
    monthly.plot(marker="o", ax=ax2)
    st.pyplot(fig2)

    # -----------------------------
    # CORRELATION HEATMAP
    # -----------------------------
    st.subheader("Feature Correlation")

    fig3, ax3 = plt.subplots()
    sns.heatmap(df[["Sales", "Estimated_Profit", "Profit_Margin"]].corr(), annot=True, ax=ax3)
    st.pyplot(fig3)

# -----------------------------
# PAGE 2 - REAL TIME PREDICTION
# -----------------------------
elif page == "🔮 Profit Prediction":

    st.title("🔮 Real-Time Profit Prediction")

    st.write("Enter business inputs below:")

    sales = st.number_input("Sales", 0.0, 10000.0, 200.0)
    discount = st.number_input("Discount", 0.0, 1.0, 0.1)
    year = st.number_input("Year", 2020, 2030, 2026)
    month = st.number_input("Month", 1, 12, 5)
    weekday = st.number_input("WeekDay", 0, 6, 2)

    if st.button("Predict Profit"):

        payload = {
            "Sales": sales,
            "Discount": discount,
            "Year": year,
            "Month": month,
            "WeekDay": weekday
        }

        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)

            result = response.json()

            st.success(f"Predicted Profit: {result['predicted_profit']:.2f}")

        except Exception as e:
            st.error(f"API Error: {e}")