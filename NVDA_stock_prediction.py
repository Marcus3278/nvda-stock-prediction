# NVIDIA Stock Price Prediction

# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch historical data for NVIDIA (NVDA)
nvda_data = yf.download('NVDA', start='2010-01-01', end='2025-04-09')

# Preprocess data
nvda_data['Date'] = nvda_data.index
nvda_data.reset_index(drop=True, inplace=True)

# Feature engineering: Create moving averages
nvda_data['MA10'] = nvda_data['Close'].rolling(window=10).mean()
nvda_data['MA50'] = nvda_data['Close'].rolling(window=50).mean()

# Drop NaN values
nvda_data.dropna(inplace=True)

# Prepare features and target
X = nvda_data[['Close', 'MA10', 'MA50']]
y = nvda_data['Close'].shift(-1)

# Remove the last row with NaN target
X = X[:-1]
y = y[:-1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title('NVIDIA Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
