# %% [markdown]
# # Algorithmic Day-Ahead BESS Optimization
# **Author:** Quantitative Developer Candidate
# 
# **Objective:** # 1. Forecast Day-Ahead wholesale electricity prices using an XGBoost regressor with time-delay embeddings and rolling volatility features.
# 2. Optimize the hyperparameters of the forecasting model using Optuna with Time-Series Cross Validation to prevent data leakage.
# 3. Feed the out-of-sample price forecast into a Mixed-Integer Linear Program (MILP) to find the profit-maximizing physical dispatch schedule for a 10 MW / 20 MWh Battery Energy Storage System (BESS).

# %%
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import pulp
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Feature Engineering & Mock Data Generation
# In production, this cell fetches data from the Energinet API. For this portfolio piece, we generate a synthetic, highly seasonal time-series reflecting the jump-diffusion and daily cyclicality of the DK2 power market.

# %%
def generate_market_data(days=365):
    """Generates synthetic hourly power prices with daily/weekly seasonality and noise."""
    np.random.seed(42)
    hours = np.arange(days * 24)
    
    # Base price + Daily cycle + Weekly cycle + Noise
    base = 300 
    daily_seasonality = 150 * np.sin(2 * np.pi * (hours - 6) / 24)
    weekly_seasonality = 50 * np.cos(2 * np.pi * hours / (24 * 7))
    noise = np.random.normal(0, 30, len(hours))
    
    prices = base + daily_seasonality + weekly_seasonality + noise
    # Simulate a few fat-tail price spikes (leptokurtic jumps)
    spike_indices = np.random.choice(hours, size=int(days*0.05), replace=False)
    prices[spike_indices] += np.random.uniform(500, 1500, size=len(spike_indices))
    
    df = pd.DataFrame({'price': prices}, index=pd.date_range("2023-01-01", periods=len(hours), freq="h"))
    return df

def create_features(df):
    """Creates time-delay embeddings and cyclical features for XGBoost."""
    df = df.copy()
    # Lags (Autoregressive components)
    df['lag_24'] = df['price'].shift(24)
    df['lag_48'] = df['price'].shift(48)
    df['lag_168'] = df['price'].shift(168) # 1 week
    
    # Rolling Statistics (Volatility/Momentum)
    df['rolling_mean_24'] = df['price'].rolling(window=24).mean()
    df['rolling_std_24'] = df['price'].rolling(window=24).std()
    
    # Cyclical Time Encodings
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df.index.dayofweek
    
    return df.dropna()

print("--- Generating Market Data ---")
df_raw = generate_market_data(days=180) # 6 months of data
df_features = create_features(df_raw)
print(f"Dataset shape after feature engineering: {df_features.shape}")

# Define features and target
features = ['lag_24', 'lag_48', 'lag_168', 'rolling_mean_24', 'rolling_std_24', 'hour_sin', 'hour_cos', 'day_of_week']
target = 'price'

X = df_features[features]
y = df_features[target]

# %% [markdown]
# ## 2. Optuna Hyperparameter Tuning
# We use Optuna to find the optimal tree depth and learning rate. Crucially, we use `TimeSeriesSplit` to respect the chronological order of the data, preventing future data from leaking into past training folds.

# %%
# Hold out the very last 24 hours as our pure out-of-sample prediction test
X_train, X_test = X.iloc[:-24], X.iloc[-24:]
y_train, y_test = y.iloc[:-24], y.iloc[-24:]

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        cv_scores.append(rmse)
        
    return np.mean(cv_scores)

print("\n--- Running Optuna Optimization ---")
# Set n_trials=10 for speed in portfolio demo; production would use 50-100
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print("Best Parameters:", study.best_params)
print(f"Best CV RMSE: {study.best_value:.2f} DKK")

# %% [markdown]
# ## 3. Final Model Training & Forecasting
# We train the final XGBoost model on the entire training set using the optimized hyperparameters, and predict the Day-Ahead auction clearing prices for tomorrow.

# %%
print("\n--- Training Final XGBoost Model ---")
best_model = xgb.XGBRegressor(**study.best_params, objective='reg:squarederror', random_state=42)
best_model.fit(X_train, y_train)

# Forecast the next 24 hours
forecasted_prices = best_model.predict(X_test)
actual_prices = y_test.values

test_rmse = np.sqrt(mean_squared_error(actual_prices, forecasted_prices))
print(f"Out-of-Sample Forecast RMSE: {test_rmse:.2f} DKK")

# Create a clean DataFrame for the optimizer
df_tomorrow = pd.DataFrame({
    'Hour': np.arange(24),
    'Forecasted_Price_DKK': np.round(forecasted_prices, 2),
    'Actual_Price_DKK': np.round(actual_prices, 2)
})
print("\nForecast Snapshot (First 5 hours):")
print(df_tomorrow.head())

# %% [markdown]
# ## 4. Mixed-Integer Linear Programming (MILP) BESS Optimization
# We take the 24-hour forecasted price vector and feed it into a PuLP solver. We introduce binary variables ($u_t, v_t$) to enforce mutual exclusivity, preventing the solver from mathematically hallucinating simultaneous charging and discharging during negative price events.

# %%
print("\n--- Running MILP BESS Optimization ---")

# Hardware Parameters (10 MW / 20 MWh Battery)
T = 24
P_max = 10.0   
E_max = 20.0   
eta_c = 0.95   
eta_d = 0.95   

prob = pulp.LpProblem("Day_Ahead_BESS_Arbitrage", pulp.LpMaximize)

# Decision Variables
C = pulp.LpVariable.dicts("Charge", range(T), lowBound=0, upBound=P_max, cat='Continuous')
D = pulp.LpVariable.dicts("Discharge", range(T), lowBound=0, upBound=P_max, cat='Continuous')
SoC = pulp.LpVariable.dicts("SoC", range(T), lowBound=0, upBound=E_max, cat='Continuous')
u = pulp.LpVariable.dicts("Is_Charging", range(T), cat='Binary')
v = pulp.LpVariable.dicts("Is_Discharging", range(T), cat='Binary')

# Objective Function: Maximize Expected Arbitrage Profit
prob += pulp.lpSum([forecasted_prices[t] * (D[t] - C[t]) for t in range(T)])

# Constraints formulation
for t in range(T):
    # 1. State of Charge Tracking
    if t == 0:
        prob += SoC[t] == 0 + (C[t] * eta_c - D[t] / eta_d) 
    else:
        prob += SoC[t] == SoC[t-1] + (C[t] * eta_c - D[t] / eta_d)
    
    # 2. Big-M linking constraints (Continuous power bound to Binary state)
    prob += C[t] <= P_max * u[t]
    prob += D[t] <= P_max * v[t]
    
    # 3. Mutual exclusivity
    prob += u[t] + v[t] <= 1

# Solve the MILP
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# %% [markdown]
# ## 5. Results & Bid Curve Extraction
# Extract the optimized schedule to construct the final bidding curve for the Nord Pool API.

# %%
print(f"Solver Status: {pulp.LpStatus[prob.status]}")
print(f"Expected Day-Ahead Profit: {pulp.value(prob.objective):,.2f} DKK\n")

print("--- Final Algorithmic Dispatch Schedule ---")
results = []
for t in range(T):
    charge_val = C[t].varValue
    discharge_val = D[t].varValue
    soc_val = SoC[t].varValue
    
    action = "IDLE"
    volume = 0.0
    if charge_val > 0.1:
        action = "BUY (Charge)"
        volume = charge_val
    elif discharge_val > 0.1:
        action = "SELL (Discharge)"
        volume = discharge_val
        
    results.append({
        "Hour": t,
        "Forecast_Price": forecasted_prices[t],
        "Action": action,
        "Volume_MW": volume,
        "SoC_MWh": soc_val
    })

df_results = pd.DataFrame(results)
# Displaying only active trading hours to save space
active_trades = df_results[df_results['Action'] != "IDLE"]
print(active_trades.to_string(index=False))