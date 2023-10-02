import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime as dt
import yfinance as yf

# Define a start date and End Date
start = dt.datetime(2011, 1, 1)
end = dt.datetime(2023, 3, 1)

st.title('Stock Trend Prediction')

stock = st.text_input('Enter Stock Ticker', 'GOOG')

df = yf.download(stock, start, end)

# Describing Data
st.subheader('Data from Jan 2011 to Mar 2023')
st.write(df.describe())

# Visualizations
ema20 = df['Close'].ewm(span=20, adjust=False).mean()
ema50 = df['Close'].ewm(span=50, adjust=False).mean()

st.subheader('Closing Price vs Time Chart with 20 & 50 Days of Exponential Moving Average')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'y')
plt.plot(ema20, 'g', label='EMA of 20 Days')
plt.plot(ema50, 'r', label='EMA of 50 Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

ema100 = df['Close'].ewm(span=100, adjust=False).mean()
ema200 = df['Close'].ewm(span=200, adjust=False).mean()

st.subheader('Closing Price vs Time Chart with 100 & 200 EMA')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.plot(ema100, 'g', label='EMA of 100 Days')
plt.plot(ema200, 'r', label='EMA of 200 Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Loading the model
model = load_model('model/my_lstm_model.h5')

# Model Testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 99:i])  # Adjusted sequence length to 99
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Plot
st.subheader('Prediction Vs Original Trend')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Plotting Future Predictions for the Next 7 Days
future_days = 7
future_predictions = []

# Use the last 100 data points as the initial input for predicting future days
input_sequence = x_test[-1]

for _ in range(future_days):
    # Predict the next day's price
    next_day_prediction = model.predict(input_sequence.reshape(1, input_sequence.shape[0], 1))
    
    # Append the predicted price to the list of future predictions
    future_predictions.append(next_day_prediction[0][0])
    
    # Update the input sequence for the next prediction by shifting and adding the new prediction
    input_sequence = np.roll(input_sequence, -1)
    input_sequence[-1] = next_day_prediction[0][0]

# Inverse transform to get the actual price values
future_predictions = np.array(future_predictions)
future_predictions = future_predictions * scale_factor

# Create a date range for the next 7 days
next_7_days = pd.date_range(start=df.index[-1] + pd.DateOffset(days=1), periods=future_days, freq='D')

# Create a DataFrame for the future predictions
future_df = pd.DataFrame({'Date': next_7_days, 'Predicted Price': future_predictions})
future_df.set_index('Date', inplace=True)

# Plot Future Predictions
st.subheader('Future Price Predictions for the Next 7 Days')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], 'g', label='Historical Price')
plt.plot(future_df.index, future_df['Predicted Price'], 'r', label='Predicted Price (Next 7 Days)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
