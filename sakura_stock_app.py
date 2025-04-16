import streamlit as st
import base64
import os
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
.st-emotion-cache-1v0mbdj {
    border: 1px solid #e94560;
    box-shadow: 0 0 15px #e94560;
}
h1 {
    color: #e94560;
    font-family: 'MS Gothic', sans-serif;
    text-shadow: 0 0 8px rgba(233,69,96,0.5);
}
</style>
""", unsafe_allow_html=True)

# 1. Configure Page
# ===== 2. REPLACE YOUR TITLE =====
st.title("📊 株価予測 / Stock Samurai")  # Hybrid Japanese-English title
st.caption("⚡ アルゴリズム武士道トレーディングシステム")  # "Algorithmic Bushido Trading System"
# ===== 3. REPLACE YOUR SIDEBAR =====
with st.sidebar:
    st.subheader("🎌 取引コントロール")  # Trading Controls
    symbol = st.text_input("銘柄コード", "AAPL")  # "Stock Code"
    days = st.slider("予測期間", 30, 90, 60, help="未来を予測する日数")  # "Days to predict"

# Samurai proverb under results
st.caption("⚔️ 武士道: 七転び八起き (Fall seven times, rise eight)")

# 2. Japanese Theme CSS
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_path = "bg.jpg"  # Image in same folder as script
img_base64 = get_img_as_base64(img_path)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# 3. Main App Function
def main():
    st.title("Stock Predictor")
    
    # Input Section
    with st.sidebar:
        st.header("Controls")
        symbol = st.text_input("Stock Symbol", "RELIANCE.NS")
        days = st.slider("Prediction Days", 30, 90, 60)
    
    # Data Fetching with Progress
    with st.spinner('Fetching stock data...'):
        try:
            data = yf.download(symbol, period="1y", progress=False)
            if data.empty:
                st.error("❌ No data found! Check your stock symbol")
                return
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            return
    
    # Convert to proper DataFrame format
    chart_data = data['Close'].reset_index()
    chart_data.columns = ['Date', 'Close']
    
    # Display Chart
    st.subheader(f"{symbol} Price History")
    st.line_chart(chart_data.set_index('Date'))
    
    # LSTM Prediction Section
    st.subheader("AI Prediction")
    if len(data) < 60:
        st.warning("⚠️ Need at least 60 days data for predictions")
        return
    
    # Prepare data for LSTM
    close_prices = data['Close'].values
    normalized = (close_prices - np.min(close_prices)) / (np.max(close_prices) - np.min(close_prices) + 1e-8)
    
    # Create sequences
    X, y = [], []
    for i in range(len(normalized) - 60):
        X.append(normalized[i:i+60])
        y.append(normalized[i+60])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Build and train model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    
    # Make prediction
    last_sequence = normalized[-60:].reshape(1, 60, 1)
    prediction = model.predict(last_sequence)
    predicted_price = prediction[0][0] * (np.max(close_prices) - np.min(close_prices)) + np.min(close_prices)
    
    st.success(f"Predicted price in {days} days: ${predicted_price:.2f}")
    # Add after creating X and y
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# Evaluate
test_predictions = model.predict(X_test)
test_mae = np.mean(np.abs(test_predictions - y_test))
print(f"Test MAE: {test_mae:.4f} (Lower = Better)")

if __name__ == "__main__":
    main()