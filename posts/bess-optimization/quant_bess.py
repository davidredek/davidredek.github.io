import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import pulp
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
import plotnine
from plotnine import ggplot, aes, geom_line, theme_bw, theme_minimal, labs, scale_y_continuous, facet_wrap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

import json # Add this at the top of your script

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
    
    if not data:
        print("No data found.")
        return pd.DataFrame()
        
    # ...so we must use 'data' here too!
    df = pd.DataFrame(data) 
    
    df = df[['TimeDK', 'DayAheadPriceDKK']].rename(
        columns={'TimeDK': 'HourDK', 'DayAheadPriceDKK': 'SpotPriceDKK'}
    )
    
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    df = df.set_index('HourDK').sort_index()
    
    return df

import pandas as pd
import requests

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
        "timezone": "Europe/Berlin"  # Matches Denmark's timezone
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

import statsmodels.api as sm
train_series.head()

 

sm.graphics.tsa.plot_acf(train_series, lags=100).show()
sm.graphics.tsa.plot_pacf(train_series, lags=48).show()

print("Training ARIMA baseline...")
# Using a simple order for compilation speed; in production, this requires ACF/PACF analysis
arima_model = ARIMA(train_series, order=(2, 1, 2))
arima_fitted = arima_model.fit()

arima_preds = arima_fitted.forecast(steps=48)
arima_mae = mean_absolute_error(test_series, arima_preds)
print(f"ARIMA Baseline MAE: {arima_mae:.2f} DKK")

def create_features(df, lags=24):
    """Generates lag and temporal features."""
    df_feat = df.copy()
    for i in range(1, lags + 1):
        df_feat[f'lag_{i}'] = df_feat['SpotPriceDKK'].shift(i)
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    return df_feat.dropna()

df_features = create_features(df_prices, lags=48)

# Features and target
X = df_features.drop(columns=['SpotPriceDKK'])
y = df_features['SpotPriceDKK']

# Train/Val/Test split (chronological)
X_train_val, X_test = X.iloc[:-48], X.iloc[-48:]
y_train_val, y_test = y.iloc[:-48], y.iloc[-48:]

# Split train_val into train and validation for Optuna
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, shuffle=False)

def objective(trial):
    """Optuna objective function for XGBoost tuning."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0)
    }
    
    model = xgb.XGBRegressor(**params, random_state=42, objective='reg:squarederror')
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

print("Running Optuna optimization for XGBoost...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15) # Kept low for execution speed

# Train final model on best params
best_params = study.best_params
print(f"Best XGBoost Params: {best_params}")

final_xgb = xgb.XGBRegressor(**best_params, random_state=42, objective='reg:squarederror')
final_xgb.fit(X_train_val, y_train_val)

xgb_preds = final_xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
print(f"XGBoost MAE: {xgb_mae:.2f} DKK")


# %%
def optimize_battery_dispatch(prices: np.ndarray, P_max=10, E_max=20, eta=0.95):
    """
    Formulates and solves the BESS linear program using PuLP.
    P_max: MW (Power capacity)
    E_max: MWh (Energy capacity)
    eta: Round-trip efficiency (assumed symmetric)
    """
    T = len(prices)
    prob = pulp.LpProblem("BESS_Dispatch", pulp.LpMaximize)
    
    # Decision Variables
    C = pulp.LpVariable.dicts("Charge", range(T), lowBound=0, upBound=P_max)
    D = pulp.LpVariable.dicts("Discharge", range(T), lowBound=0, upBound=P_max)
    SoC = pulp.LpVariable.dicts("SoC", range(T), lowBound=0, upBound=E_max)
    
    # Objective: Maximize Arbitrage Profit
    # Profit = Revenue from discharging - Cost of charging
    prob += pulp.lpSum([prices[t] * (D[t] - C[t]) for t in range(T)])
    
    # Constraints
    for t in range(T):
        if t == 0:
            # Assume battery starts empty
            prob += SoC[t] == C[t] * eta - (D[t] / eta)
        else:
            # Transition state
            prob += SoC[t] == SoC[t-1] + C[t] * eta - (D[t] / eta)
            
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    # Results formatting
    profit = pulp.value(prob.objective)
    schedule = pd.DataFrame({
        'Forecasted_Price': prices,
        'Charge_MW': [C[t].varValue for t in range(T)],
        'Discharge_MW': [D[t].varValue for t in range(T)],
        'SoC_MWh': [SoC[t].varValue for t in range(T)]
    })
    
    return profit, schedule

print("Running BESS Optimization using XGBoost predictions...")
# Use the XGBoost predictions (last 48 hours) for the optimization
forecasted_prices = xgb_preds
optimal_profit, dispatch_schedule = optimize_battery_dispatch(forecasted_prices)

print(f"\nExpected Optimal Profit for 48h window: {optimal_profit:.2f} DKK")
print("\nFirst 5 hours of Dispatch Schedule:")
print(dispatch_schedule.head())




from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
from plotnine.data import mtcars

a = (ggplot(mtcars, aes('wt', 'mpg', color='factor(gear)'))
 + geom_point()
 + stat_smooth(method='lm')
 + facet_wrap('~gear'))

print(a)