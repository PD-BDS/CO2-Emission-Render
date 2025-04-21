import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os

st.set_page_config(
    page_title="CO₂ Emission Forecasting Dashboard",
    page_icon="🌍",
    layout="wide"
)

API = os.getenv("BACKEND_URL", "http://localhost:8000/api")

st.title("🌍 CO₂ Emission Forecasting Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Last 24h Emissions",
    "Next 6h Forecast",
    "Recent Predictions vs Actual",
    "Model Info"
])

def safe_get(endpoint):
    try:
        r = requests.get(f"{API}{endpoint}")
        if r.status_code == 200:
            data = r.json()
            # Ensure list of records
            return data if isinstance(data, list) else [data]
        else:
            st.warning(f"API call failed: {r.status_code}")
            return []
    except Exception as e:
        st.error(f"Error contacting backend: {e}")
        return []

# 📊 Tab 1: Last 24h CO2 Emissions
with tab1:
    df = pd.DataFrame(safe_get("/last-24h-emissions"))
    if df.empty or "TimeStamp" not in df.columns or "CO2Emission" not in df.columns:
        st.warning("No valid CO₂ emission data available.")
    else:
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
        df = df.sort_values("TimeStamp")

        hours = st.slider("Select hours to display", min_value=6, max_value=24, value=24, step=2)
        df = df.tail(hours)
        fig = px.line(df, x="TimeStamp", y="CO2Emission", title=f"Last {hours} Hours CO₂ Emissions")
        fig.update_traces(mode="lines+markers+text", text=df["CO2Emission"], textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

# 🔮 Tab 2: Next 6h Forecast
with tab2:
    df = pd.DataFrame(safe_get("/next-6h-predictions"))
    if df.empty or "TimeStamp" not in df.columns or "Prediction" not in df.columns:
        st.warning("No forecast data available.")
    else:
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
        df["Prediction"] = round(df["Prediction"], 2)
        fig = px.line(df, x="TimeStamp", y="Prediction", title="Next 6 Hours Forecast")
        fig.update_traces(mode="lines+markers+text", text=df["Prediction"], textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

# 🔍 Tab 3: Prediction vs Actual (last 6h)
with tab3:
    df = pd.DataFrame(safe_get("/last-6h-predictions-vs-actual"))
    if df.empty or "TimeStamp" not in df.columns or "Prediction" not in df.columns:
        st.warning("No recent prediction data.")
    else:
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
        df["Prediction"] = round(df["Prediction"], 2)
        df["Actual"] = df.get("Actual", pd.Series([None]*len(df)))
        df["Actual"] = df["Actual"].fillna(0, inplace=True)

        fig = px.line(df.melt(id_vars="TimeStamp", value_vars=["Prediction", "Actual"]),
                      x="TimeStamp", y="value", color="variable",
                      title="Prediction vs Actual (Last 6h)", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# 🧠 Tab 4: Best Model Info
with tab4:
    model_data = safe_get("/best-model")
    if model_data and isinstance(model_data, list) and isinstance(model_data[0], dict):
        model = model_data[0]
        st.metric("Model", model.get("Model_name", "N/A"))
        st.metric("Version", model.get("Version", "N/A"))
        st.metric("Accuracy", f"{model.get('Pseudo_accuracy', 0):.2f}%")
        st.metric("RMSE", f"{model.get('RMSE', 0):.2f}")
        st.metric("MAPE", f"{model.get('MAPE', 0):.2f}")
        st.metric("R²", f"{model.get('R2', 0):.2f}")
    else:
        st.warning("Model metadata unavailable.")
