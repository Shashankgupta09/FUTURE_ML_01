# cleaned_script.py - the model pipeline (no data generation)
import pandas as pd
from prophet import Prophet

def run_forecast(csv_path, periods=365):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.rename(columns={'Date':'ds','Sales':'y'})[['ds','y']].sort_values('ds')
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

if __name__ == "__main__":
    print("Run run_forecast('sales_sample.csv')")