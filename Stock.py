# stock_predictor.py

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------------------
# CONFIGURATION
# ---------------------------
TICKER = "AAPL"       # 👈 Change to your stock (e.g., TSLA, MSFT, GOOGL)
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
FUTURE_DAYS = 30      # Predict next 30 days

# ---------------------------
# 1. Fetch Stock Data
# ---------------------------
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
print(df.head())

# Use only closing price
data = df[["Close"]].values

# ---------------------------
# 2. Preprocess Data
# ---------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ---------------------------
# 3. Build LSTM Model
# ---------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

# ---------------------------
# 4. Predictions
# ---------------------------
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# ---------------------------
# 5. Plot Results
# ---------------------------
plt.figure(figsize=(14, 6))
plt.plot(df.index, data, label="Actual Price", color="blue")

# Training prediction plot
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict)+time_step, :] = train_predict
plt.plot(df.index, train_plot, label="Train Predict", color="green")

# Testing prediction plot
test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(train_predict)+(time_step*2)+1:len(data)-1, :] = test_predict
plt.plot(df.index, test_plot, label="Test Predict", color="red")

plt.title(f"{TICKER} Stock Price Prediction with LSTM")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# ---------------------------
# 6. Future Prediction
# ---------------------------
last_60 = scaled_data[-60:]
future_input = last_60.reshape(1, -1, 1)

predicted_prices = []
for _ in range(FUTURE_DAYS):
    pred = model.predict(future_input)[0][0]
    predicted_prices.append(pred)
    future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Plot future forecast
plt.figure(figsize=(10, 5))
plt.plot(range(FUTURE_DAYS), predicted_prices, color="orange", label="Future Prediction")
plt.title(f"{TICKER} Next {FUTURE_DAYS} Days Forecast")
plt.xlabel("Days Ahead")
plt.ylabel("Predicted Price")
plt.legend()
plt.show()
