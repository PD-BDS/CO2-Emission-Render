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

API = os.getenv("BACKEND_URL", "/api")

st.title("🌍CO₂ Emission Forecasting Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Last 24h Emissions", "Next 6h Forecast", "Recent Predictions vs Actual", "Model Info"])

with tab1:
    r = requests.get(f"{API}/last-24h-emissions")
    df = pd.DataFrame(r.json())
    if df.empty:
        st.warning("No data available.")
    else:
        hours = st.slider("Select hours to display", min_value=6, max_value=24, value=24, step=2)
        df = df.tail(hours)
        fig = px.line(df, x="TimeStamp", y="CO2Emission", title=f"Last {hours} Hours CO₂ Emissions")
        fig.update_traces(mode="lines+markers+text", text=df["CO2Emission"], textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    r = requests.get(f"{API}/next-6h-predictions")
    df = pd.DataFrame(r.json())
    
    if df.empty or "TimeStamp" not in df.columns:
        st.warning("No data available for the last 24 hours.")
    else:
        df["Prediction"] = round(df["Prediction"], 2)
        fig = px.line(df, x="TimeStamp", y="Prediction", title="Next 6 Hours Forecast")
        fig.update_traces(mode="lines+markers+text", text=df["Prediction"], textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    r = requests.get(f"{API}/last-6h-predictions-vs-actual")
    
    if r.status_code == 200:
        df = pd.DataFrame(r.json())

        if df.empty or "TimeStamp" not in df.columns:
            st.warning("No data available for the last 6 hours.")
        else:
            df["Prediction"] = round(df["Prediction"], 2)
            df['Actual'].fillna(0, inplace=True)
            fig = px.line(df.melt(id_vars="TimeStamp", value_vars=["Prediction", "Actual"]),
                          x="TimeStamp", y="value", color="variable", title="Prediction vs Actual (Last 6h)", markers=True)
            
            
            # Plot the chart
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Failed to retrieve data. Status code: {r.status_code}")


with tab4:
    r = requests.get(f"{API}/best-model")
    model = r.json()[0]
    st.metric("Model", model['Model_name'])
    st.metric("Version", model['Version'])
    st.metric("Accuracy", f"{model['Pseudo_accuracy']:.2f}%")
    st.metric("RMSE", f"{model['RMSE']:.2f}")
    st.metric("MAPE", f"{model['MAPE']:.2f}")
    st.metric("R²", f"{model['R2']:.2f}")
