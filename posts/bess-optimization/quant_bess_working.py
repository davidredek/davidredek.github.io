import requests 
import pandas as pd
import polars as pl
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from plotnine import ggplot, aes, geom_line, theme_bw, geom_point, theme_minimal, labs, scale_y_continuous, facet_wrap
import matplotlib.pyplot as plt
import matplotlib


def fetch_dk2_spot_prices(start_date: str, end_date: str) -> pd.DataFrame:
    url = 'https://api.energidataservice.dk/dataset/DayAheadPrices'
    
    params = {
        'start': start_date.replace(' ', 'T'),
        'end': end_date.replace(' ', 'T'),
        'filter': '{"PriceArea":["DK2"]}', 
        'sort': 'TimeDK ASC',
        'limit': 10000 
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    # We assigned the API result to 'data' here...
    data = response.json().get('records', [])
    
    df = pd.DataFrame(data) 
    
    df = df[['TimeDK', 'DayAheadPriceDKK']].rename(
        columns={'TimeDK': 'HourDK', 'DayAheadPriceDKK': 'SpotPriceDKK'}
    )
    
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    df = df.set_index('HourDK').sort_index()
    
    return df
a=requests.get("https://archive-api.open-meteo.com/v1/archive",     params = {
        "latitude": 55.6761,
        "longitude": 12.5683,
        "start_date": "2025-12-01T00:00".split('T')[0],
        "end_date": "2026-03-01T00:00".split('T')[0],
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "Europe/Berlin"  # Matches Denmark's timezone
    }).json().get('hourly', {})

a.keys()
a['time']
d = pl.DataFrame({
    "HourDK": a['time'],
    "Temp": a['temperature_2m'],
    "WindSpeed": a['wind_speed_10m'],
    "Solar": a['shortwave_radiation']
}).with_columns(pl.col("HourDK").str.to_datetime())

p = (
    ggplot(data = d, mapping = aes(x = 'HourDK', y = 'Temp')) + geom_line() + theme_bw()
)

p.show()

def fetch_dk2_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical/forecast weather data for Copenhagen (DK2 proxy).
    Variables: Temperature, Wind Speed (10m), and Solar Radiation.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Coordinates for Copenhagen (representing DK2)
    params = {
        "latitude": 55.6761,
        "longitude": 12.5683,
        "start_date": start_date.split('T')[0],
        "end_date": end_date.split('T')[0],
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "Europe/Berlin"  
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    res_data = response.json()
    
    # Extract the hourly dictionary
    hourly = res_data.get('hourly', {})
    
    # Create DataFrame
    df_weather = pd.DataFrame({
        "HourDK": pd.to_datetime(hourly.get("time")),
        "Temp_C": hourly.get("temperature_2m"),
        "WindSpeed_ms": hourly.get("wind_speed_10m"),
        "Solar_Wm2": hourly.get("shortwave_radiation")
    })
    
    return df_weather

df_weather = fetch_dk2_weather("2025-12-01T00:00", "2026-03-01T00:00")
df_price = fetch_dk2_spot_prices("2025-12-01T00:00", "2026-03-01T00:00").resample('h').mean() # Resample to hourly and forward-fill missing values

df = df_price.join(df_weather.set_index('HourDK'), how='left').reset_index()



(
    ggplot(df.reset_index().melt(id_vars='HourDK', value_vars=['SpotPriceDKK', 'Temp_C', 'WindSpeed_ms', 'Solar_Wm2'], var_name='Variable', value_name='Value'), 
           aes(x='HourDK', y='Value', color='Variable')) +
    geom_line() +
    facet_wrap('~Variable', scales='free_y') +
    theme_bw()
).show()

# Splitting the last 48 hours as our test set for the day-ahead + 1 scenario
train_series = df['SpotPriceDKK'].iloc[:-48]
test_series = df['SpotPriceDKK'].iloc[-48:]
