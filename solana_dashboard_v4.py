# solana_dashboard_v4.py
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import argparse
import os
import matplotlib.pyplot as plt
import time
import streamlit as st
from datetime import datetime

# ----------------------------------------------------
# TECHNICAL INDICATORS
# ----------------------------------------------------
def rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ----------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------
def create_features(df, lags=5, horizon=12):
    """Create time-series features from a dataframe."""
    df = df.copy()
    # Lagged features for Close
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    # Rolling window features
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['diff_1'] = df['Close'].diff(1)
    # New: Volatility
    df['vol_5'] = df['Close'].rolling(window=5).std()
    # New: Technical indicators
    df['rsi_14'] = rsi(df['Close'], 14)
    macd_line, macd_signal, _ = macd(df['Close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    # New: Volume features
    df['volume_lag_1'] = df['Volume'].shift(1)
    df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
    # Multi-step future targets
    for h in range(1, horizon + 1):
        df[f'target_{h}'] = df['Close'].shift(-h)
    df = df.dropna()
    return df

# ----------------------------------------------------
# DATA FETCHING
# ----------------------------------------------------
def fetch_data(symbol, period, interval):
    """Fetches data for a given symbol, period, and interval."""
    print(f"Fetching {interval} data for {symbol} over the last {period}...")
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        if df.empty:
            print(f"Warning: No data fetched for {interval}. Skipping.")
            return None
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching data for {interval}: {str(e)}")
        return None

# ----------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------
def train_model(df, timeframe, horizon=12):
    """Trains an XGBoost model for a given timeframe and saves it."""
    df_feat = create_features(df, horizon=horizon)
    X = df_feat.drop([f'target_{h}' for h in range(1, horizon + 1)] + ['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    y = df_feat[[f'target_{h}' for h in range(1, horizon + 1)]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    models = []
    print(f"\nTraining models for {timeframe} timeframe...")
    for i, col in enumerate(y_train.columns):
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train[col])
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test[col], y_pred)
        print(f"  MAE for {col}: {mae:.4f}")
        models.append(model)
    model_filename = f"{timeframe}_model.pkl"  # Updated to support any symbol by removing 'sol_'
    joblib.dump(models, model_filename)
    print(f"\nSaved multi-step model as {model_filename}")
    return models

# ----------------------------------------------------
# PREDICT NEXT 12 CANDLES
# ----------------------------------------------------
def predict_next_12_candles(models, df_feat, horizon=12):
    """Makes predictions for the next 12 candles."""
    last_row = df_feat.drop([f'target_{h}' for h in range(1, horizon + 1)] + ['Open', 'High', 'Low', 'Close', 'Volume'], axis=1).iloc[-1]
    preds = []
    for model in models:
        p = model.predict(last_row.values.reshape(1, -1))[0]
        preds.append(p)
    return preds

# ----------------------------------------------------
# PLOT PREDICTIONS
# ----------------------------------------------------
def plot_predictions(df, preds, timeframe):
    """Generates a Matplotlib plot of past data and future predictions."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot historical data (last 48 candles) on ax
    ax.plot(df.index, df['Close'], label='Historical Close', color='gray', alpha=0.5)
    ax.plot(df.tail(48).index, df['Close'].tail(48), label='Past 48 Candles', marker='o', color='blue', linestyle='-')
    
    # Generate future timestamps
    last_time = df.index[-1]
    # Use the timeframe string directly for frequency
    freq_map = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1d'}
    freq = freq_map.get(timeframe, timeframe)  # Map to pandas frequency string
    future_times = pd.date_range(start=last_time, periods=len(preds) + 1, freq=freq)[1:]
    
    ax.plot(future_times, preds, label='Predicted Next 12 Candles', marker='x', linestyle='--', color='red')
    ax.set_title(f"{symbol} Price Prediction ({timeframe})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    
    # Zoom in on the recent past and future
    view_start_time = df.index[-48] if len(df) >= 48 else df.index[0]
    view_end_time = future_times[-1]
    ax.set_xlim(view_start_time, view_end_time)
    
    # Volume subplot on ax2
    ax2.bar(df.tail(48).index, df['Volume'].tail(48), label='Volume', color='green', alpha=0.3)
    ax2.set_title("Volume (Last 48 Candles)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Volume")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(view_start_time, df.index[-1])
    
    fig.tight_layout()
    return fig

# ----------------------------------------------------
# LIVE DASHBOARD (Streamlit)
# ----------------------------------------------------
def run_dashboard(symbol="SOL-USD", refresh_rate=300):
    st.set_page_config(page_title=f"{symbol} Prediction Dashboard", layout="wide")
    st.title(f"📈 {symbol} Live Prediction Dashboard")

    # New: Expanded timeframes
    timeframes = {
        '5m': '5d',
        '15m': '1mo',  # New timeframe
        '1h': '2mo',
        '4h': '6mo',
        '1d': '1y',    # New timeframe
    }
    placeholder = st.empty()

    # New: Sidebar for settings
    with st.sidebar:
        st.header("Dashboard Settings")
        refresh_rate = st.slider("Refresh Rate (seconds)", min_value=60, max_value=1800, value=refresh_rate, step=60)
        retrain_button = st.button("Retrain All Models")

    while True:
        if retrain_button:
            for interval in timeframes:
                df = fetch_data(symbol, timeframes[interval], interval)
                if df is not None:
                    with st.spinner(f"Retraining {interval} model..."):
                        train_model(df, interval)
            retrain_button = False  # Reset after retrain

        with placeholder.container():
            # New: Current Price Display
            current_df = fetch_data(symbol, '1d', '1m')  # Fetch recent data for current price
            if current_df is not None:
                current_price = float(current_df['Close'].iloc[-1])
                st.metric("Current Price", f"${current_price:.2f}")

            st.header("Multi-Timeframe Analysis")
            cols = st.columns(len(timeframes))
            
            for i, (interval, period) in enumerate(timeframes.items()):
                with cols[i]:
                    st.subheader(f"{interval.upper()} Chart")
                    model_filename = f"{interval}_model.pkl"

                    df = fetch_data(symbol, period, interval)
                    if df is None or df.empty:
                        st.warning(f"Could not fetch data for {interval}.")
                        continue
                    
                    if not os.path.exists(model_filename):
                        with st.spinner(f"Training {interval} model..."):
                            train_model(df, interval)
                    
                    models = joblib.load(model_filename)
                    df_feat = create_features(df)
                    predictions = predict_next_12_candles(models, df_feat)

                    # --- Trend Indicator Logic ---
                    last_known_price = float(df['Close'].iloc[-1]) 
                    first_prediction = float(predictions[0])
                    price_diff = first_prediction - last_known_price
                    
                    if price_diff > 0:
                        trend = "UP"
                        delta_color = "normal"
                        emoji = "🔼"
                    else:
                        trend = "DOWN"
                        delta_color = "inverse"
                        emoji = "🔽"
                        
                    st.metric(
                        label=f"Prediction for next candle: {trend} {emoji}",
                        value=f"${first_prediction:.4f}",
                        delta=f"{price_diff:+.4f} USD",
                        delta_color=delta_color
                    )
                    
                    fig = plot_predictions(df, predictions, interval)
                    st.pyplot(fig)

                    # New: Predictions Table
                    pred_df = pd.DataFrame({
                        'Candle': [f"t+{i}" for i in range(1, 13)],
                        'Predicted Price': [f"${p:.4f}" for p in predictions]
                    })
                    st.table(pred_df)

            st.success(f"Dashboard updated successfully. Next refresh in {refresh_rate} seconds. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        time.sleep(refresh_rate)

# ----------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Crypto Price Prediction Tool")  # Updated description
    parser.add_argument('--mode', type=str, default='dashboard', choices=['train', 'predict', 'dashboard'], help='Operation mode')
    parser.add_argument('--symbol', type=str, default='SOL-USD', help='Ticker symbol (e.g., SOL-USD, BTC-USD)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (e.g., 5m, 15m, 1h, 4h, 1d)')
    args = parser.parse_args()

    global symbol  # Make symbol global for use in functions
    symbol = args.symbol

    if args.mode == 'dashboard':
        run_dashboard(symbol=args.symbol)
        return

    timeframe_map = {'5m': '5d', '15m': '1mo', '1h': '2mo', '4h': '6mo', '1d': '1y'}
    if args.timeframe not in timeframe_map:
        print(f"Error: Invalid timeframe. Choose from {list(timeframe_map.keys())}")
        return

    period = timeframe_map[args.timeframe]
    df = fetch_data(args.symbol, period, args.timeframe)
    if df is None: return

    if args.mode == 'train':
        train_model(df, args.timeframe)
    elif args.mode == 'predict':
        model_filename = f"{args.timeframe}_model.pkl"
        if not os.path.exists(model_filename):
            print(f"Model '{model_filename}' not found. Run training first.")
            return
        models = joblib.load(model_filename)
        df_feat = create_features(df)
        preds = predict_next_12_candles(models, df_feat)
        print("\n📈 --- NEXT 12 CANDLES PREDICTION ---")
        for i, p in enumerate(preds, 1):
            print(f"t+{i} -> ${p:.4f}")
        fig = plot_predictions(df, preds, args.timeframe)
        plt.show()

if __name__ == "__main__":
    main()
