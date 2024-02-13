import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Fetch daily stock prices for the year 2022-2023
stock_data = yf.download('AAPL', start='2022-01-01', end='2023-01-01', progress=False)

#for Handle missing values 
stock_data = stock_data.dropna()

# We'll use Closing price for our analysis and prediction
prices = stock_data['Close'].values

# Calculate moving average
def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'valid') / period

# Define a period for moving average
period = 5
moving_avg = moving_average(prices, period)

# Prepare features for SVM (prices and moving average)
# Align lengths of features, trim data to match moving average array length
features = np.column_stack((prices[period-1:], moving_avg))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, prices[period-1:], test_size=0.2, random_state=42)

# Scale the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use PCA for dimensionality reduction
pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train the model with PCA-transformed data
model = SVR(kernel='rbf')
model.fit(X_train_pca, y_train)

# Predict with PCA-transformed data
predictions_pca = model.predict(X_test_pca)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions_pca)
print(f'Mean Squared Error: {mse}')

# we plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test, color='black', label='Actual prices')
plt.plot(predictions_pca, color='blue', label='Predicted prices with PCA')
plt.title('Actual prices vs Predicted prices with PCA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
