# Full Fixed Version — Live Price + Zoomed Graphs + No Crashes on Free Tier
# Works on Streamlit Community Cloud (ephemeral storage)

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ----------------------------------------------------
# TECHNICAL INDICATORS
# ----------------------------------------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
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
    df = df.copy()

    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['diff_1'] = df['Close'].diff(1)
    df['vol_5'] = df['Close'].rolling(window=5).std()

    df['rsi_14'] = rsi(df['Close'], 14)
    macd_line, macd_signal, _ = macd(df['Close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal

    df['volume_lag_1'] = df['Volume'].shift(1)
    df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()

    for h in range(1, horizon + 1):
        df[f'target_{h}'] = df['Close'].shift(-h)

    return df.dropna()

# ----------------------------------------------------
# DATA FETCHING
# ----------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        if df.empty:
            return None
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None

# ----------------------------------------------------
# MODEL TRAINING (Faster + Safer for Free Tier)
# ----------------------------------------------------
def train_model(df, timeframe, horizon=12):
    df_feat = create_features(df, horizon=horizon)

    X = df_feat.drop([f'target_{h}' for h in range(1, horizon + 1)] +
                     ['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)

    y = df_feat[[f'target_{h}' for h in range(1, horizon + 1)]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    models = []
    for col in y_train.columns:
        model = XGBRegressor(
            n_estimators=100,           # Reduced from 300 → fast & safe
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,                   # Critical: 1 job on free tier
            tree_method='hist'
        )
        model.fit(X_train, y_train[col])
        models.append(model)

    # Save safely
    model_file = f"{timeframe}_model.pkl"
    try:
        joblib.dump(models, model_file)
    except:
        pass  # Ignore save errors — will retrain next time
    return models

# ----------------------------------------------------
# FORECASTING
# ----------------------------------------------------
def predict_next_12_candles(models, df_feat, horizon=12):
    last_row = df_feat.drop([f'target_{h}' for h in range(1, horizon + 1)] +
                             ['Open', 'High', 'Low', 'Close', 'Volume'], axis=1).iloc[-1]
    return [model.predict(last_row.values.reshape(1, -1))[0] for model in models]

# ----------------------------------------------------
# PLOTTING (ZOOM FIXED + CLEARER)
# ----------------------------------------------------
def plot_predictions(df, preds, timeframe, symbol):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]})

    ax.plot(df.index, df['Close'], label='Historical', color='gray', alpha=0.4)
    ax.plot(df.tail(48).index, df['Close'].tail(48), marker='o', label='Last 48')

    freq_map = {'5m': '5min', '15m': '15min', '1h': 'H', '4h': '4H', '1d': 'D'}
    future_times = pd.date_range(df.index[-1], periods=len(preds) + 1, freq=freq_map[timeframe])[1:]

    ax.plot(future_times, preds, marker='x', linestyle='--', label='Prediction', color='red')

    # ---- ZOOM FIX ----
    if len(df) >= 48:
        view_start = df.index[-48]
    else:
        view_start = df.index[0]

    ax.set_xlim(view_start, future_times[-1])
    ax.set_title(f"{symbol} Prediction ({timeframe})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2.bar(df.tail(48).index, df['Volume'].tail(48), color='lightblue', alpha=0.7)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Volume (Last 48 Candles)")

    plt.close(fig)  # Prevent memory leak
    return fig

# ----------------------------------------------------
# SAFE MODEL LOADER (No EOFError on Free Tier)
# ----------------------------------------------------
def load_or_train_model(df, timeframe):
    model_file = f"{timeframe}_model.pkl"
    models = None

    if os.path.exists(model_file):
        try:
            if os.path.getsize(model_file) > 1000:  # Not empty
                models = joblib.load(model_file)
                st.toast(f"Loaded {timeframe.upper()} model", icon="Loaded")
        except Exception as e:
            st.warning(f"Model corrupted, retraining {timeframe.upper()}...")

    if models is None:
        with st.spinner(f"Training {timeframe.upper()} model..."):
            models = train_model(df, timeframe)
        st.toast(f"{timeframe.upper()} model trained!", icon="Trained")

    return models

# ----------------------------------------------------
# STREAMLIT DASHBOARD
# ----------------------------------------------------
def run_dashboard(symbol="SOL-USD", refresh_rate=300):
    st.set_page_config(page_title=f"{symbol} Dashboard", layout="wide")
    st.title(f"{symbol} Prediction Dashboard")

    # Auto-refresh
    st_autorefresh(interval=refresh_rate * 1000, key="refresh")

    # ---- LIVE PRICE ----
    current_df = fetch_data(symbol, '1d', '1m')
    if current_df is not None and len(current_df) > 0:
        live_price = float(current_df['Close'].iloc[-1])
        st.metric("Live Price", f"${live_price:.4f}", delta=None)
    else:
        st.warning("Live price unavailable")

    timeframes = {
        '5m': '5d',
        '15m': '1mo',
        '1h': '2mo',
        '4h': '6mo',
        '1d': '1y'
    }

    with st.sidebar:
        st.header("Settings")
        refresh_rate = st.slider("Refresh rate (sec)", 60, 1800, refresh_rate, 60)
        retrain = st.button("Retrain All Models")

    if retrain:
        for tf, period in timeframes.items():
            df = fetch_data(symbol, period, tf)
            if df is not None:
                with st.spinner(f"Retraining {tf.upper()}..."):
                    train_model(df, tf)
        st.success("All models retrained!")

    cols = st.columns(len(timeframes))

    for i, (tf, period) in enumerate(timeframes.items()):
        with cols[i]:
            st.subheader(tf.upper())

            df = fetch_data(symbol, period, tf)
            if df is None or df.empty:
                st.warning("No data.")
                continue

            models = load_or_train_model(df, tf)
            df_feat = create_features(df)
            preds = predict_next_12_candles(models, df_feat)

            last_price = float(df['Close'].iloc[-1])
            diff = preds[0] - last_price

            st.metric(
                "Next Candle", f"${preds[0]:.4f}", f"{diff:+.4f}",
                delta_color="normal" if diff > 0 else "inverse"
            )

            fig = plot_predictions(df, preds, tf, symbol)
            st.pyplot(fig)

            table = pd.DataFrame({
                "Candle": [f"t+{i}" for i in range(1, 13)],
                "Prediction": [f"${p:.4f}" for p in preds]
            })
            st.dataframe(table, use_container_width=True)

# ----------------------------------------------------
# MAIN — Streamlit Cloud Ready
# ----------------------------------------------------
if __name__ == "__main__":
    run_dashboard(symbol="SOL-USD")
