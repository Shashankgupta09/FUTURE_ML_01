
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import altair as alt

st.set_page_config(page_title="AI Sales Forecast Dashboard", layout="wide")
st.title("AI Sales Forecasting Dashboard")

st.sidebar.header("Upload data")
uploaded = st.sidebar.file_uploader("Upload CSV with 'Date' and 'Sales' columns", type=["csv"])

if uploaded is None:
    st.sidebar.info("No file uploaded — using provided sample dataset.")
    df = pd.read_csv("sales_sample.csv", parse_dates=["Date"])
else:
    df = pd.read_csv(uploaded, parse_dates=["Date"])

st.sidebar.header("Forecast settings")
periods = st.sidebar.number_input("Forecast days", min_value=30, max_value=365*5, value=365)
seasonality_mode = st.sidebar.selectbox("Seasonality mode", ["additive","multiplicative"])
changepoint = st.sidebar.slider("Changepoint prior scale", 0.001, 0.5, 0.05)

# Prepare data
df = df.sort_values("Date")
df_prophet = df.rename(columns={"Date":"ds", "Sales":"y"})[["ds","y"]]

# Train model
with st.spinner("Training Prophet model..."):
    model = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=changepoint)
    model.fit(df_prophet)

future = model.make_future_dataframe(periods=periods)
forecast = model.predict(future)

# KPI row
col1, col2, col3 = st.columns(3)
col1.metric("Last recorded sales", int(df['Sales'].iloc[-1]))
col2.metric("Average daily sales (historical)", int(df['Sales'].mean()))
col3.metric("Forecast horizon (days)", periods)

# Main chart
st.subheader("Historical Sales and Forecast")
chart_df = pd.concat([df_prophet.set_index("ds")["y"], forecast.set_index("ds")["yhat"]], axis=1)
chart_df = chart_df.reset_index().melt(id_vars="ds", var_name="series", value_name="value")
base = alt.Chart(chart_df).mark_line().encode(
    x='ds:T',
    y='value:Q',
    color='series:N'
).properties(width=900, height=400)
st.altair_chart(base, use_container_width=True)

# Show forecast table
st.subheader("Forecast (next 30 rows)")
st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(30))

# Components
st.subheader("Forecast components")
fig = model.plot_components(forecast)
st.pyplot(fig)
