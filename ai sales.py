import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
from datetime import date, timedelta

# ------------------- CONFIG -------------------
FORECAST_PERIOD_DAYS = 365 * 2  # Forecast 2 years ahead
START_DATE = date(2018, 1, 1)
END_DATE = date(2022, 12, 31)

print("1. Generating mock historical retail sales data...")

# ------------------- 1. DATA GENERATION -------------------
# Create daily date range
date_range = pd.date_range(start=START_DATE, end=END_DATE)
df = pd.DataFrame({'Date': date_range})

# Trend (linear increase)
df['Trend'] = np.arange(len(df)) * 5

# Yearly seasonality (peak during festival/holiday season)
df['Yearly_Seasonality'] = 1000 * np.sin(df['Date'].dt.dayofyear / 365.25 * 2 * np.pi)

# Weekly seasonality (weekend boost)
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0
df['Weekly_Seasonality'] = np.where(df['DayOfWeek'].isin([5, 6]), 500, 0)

# Random noise
np.random.seed(42)
df['Noise'] = np.random.normal(loc=0, scale=300, size=len(df))

# Final sales = trend + seasonality + constant + noise
df['Sales'] = (
    df['Trend']
    + df['Yearly_Seasonality']
    + df['Weekly_Seasonality']
    + 10000
    + df['Noise']
)

df['Sales'] = df['Sales'].clip(lower=0).astype(int)

print(f"Generated historical data from {START_DATE} to {END_DATE} ({len(df)} days).")

# ------------------- 2. PREPARING DATA FOR PROPHET -------------------
df_prophet = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
print("Prepared dataset for Prophet (ds, y).")

# ------------------- 3. MODEL TRAINING -------------------
model = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

model.fit(df_prophet)
print("Prophet model training complete.")

# ------------------- 4. FORECASTING -------------------
future = model.make_future_dataframe(periods=FORECAST_PERIOD_DAYS)
forecast = model.predict(future)
print(f"Forecast created for next {FORECAST_PERIOD_DAYS} days.")

# ------------------- 5. VISUALIZATION -------------------

# A. Main forecast plot
model.plot(forecast, xlabel='Date', ylabel='Sales', figsize=(12, 6))
plt.title("AI Sales Forecast: Trend + Seasonality + Future Projection")
plt.show()

# B. Component breakdown: Trend + Yearly + Weekly seasonality
model.plot_components(forecast, figsize=(12, 10))
plt.suptitle('Forecast Components (Trend / Yearly / Weekly)', y=1.02)
plt.show()

# C. Zoomed recent-year forecast
recent_data = df_prophet[df_prophet['ds'] > pd.to_datetime(END_DATE) - timedelta(days=365)]
forecast_recent = forecast[forecast['ds'] > pd.to_datetime(END_DATE) - timedelta(days=365 * 2)]

plt.figure(figsize=(12, 6))
plt.plot(recent_data['ds'], recent_data['y'], 'k.', label='Historical Sales (Last Year)')
plt.plot(forecast_recent['ds'], forecast_recent['yhat'], label='Forecast')

plt.fill_between(
    forecast_recent['ds'],
    forecast_recent['yhat_lower'],
    forecast_recent['yhat_upper'],
    alpha=0.2,
    label='Confidence Interval'
)

plt.axvline(pd.to_datetime(END_DATE), color='red', linestyle='--', label='Forecast Start')
plt.title("Recent History + Near-Term Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

print("\nPipeline execution complete. All visualizations generated.")

# ------------------- 6. SAMPLE FORECAST OUTPUT -------------------
print("\n--- Sample Forecast (last 5 days) ---")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
