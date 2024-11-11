## DEVELOPED BY: M JAYACHANDRAN
## REGISTER NO: 212222240038
## DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('future_gold_price.csv')

# Convert 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use 'Close' price as the target variable, removing commas and converting to numeric
data['Close'] = data['Close'].replace(',', '', regex=True).astype(float)

# Resample data to monthly frequency, taking the mean of 'Close' prices per month
monthly_data = data['Close'].resample('M').mean().dropna()

# Check for stationarity using the Augmented Dickey-Fuller (ADF) test
result = adfuller(monthly_data)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets (80% training, 20% testing)
train_data = monthly_data.iloc[:int(0.8 * len(monthly_data))]
test_data = monthly_data.iloc[int(0.8 * len(monthly_data)):]

# Define the lag order for the AutoRegressive model based on seasonality
lag_order = 12  # Monthly lag for seasonal data
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

# Plot Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(monthly_data, lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Monthly Gold Prices')
plt.show()

# Plot Partial Autocorrelation Function (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(monthly_data, lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Monthly Gold Prices')
plt.show()

# Make predictions on the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate Mean Squared Error (MSE) for predictions
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

# Plot Test Data vs Predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Test Data - Monthly Gold Prices', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Monthly Gold Prices', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.title('AR Model Predictions vs Test Data (Monthly Gold Prices)')
plt.legend()
plt.grid(True)
plt.show()

# Forecast Future Prices
forecast_steps = 10  # Number of months to forecast
future_predictions = model_fit.predict(start=len(monthly_data), end=len(monthly_data) + forecast_steps - 1)

# Create a date range for future predictions
future_dates = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# Plot historical prices, test predictions, and future forecasts
plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, monthly_data, label='Historical Prices')
plt.plot(test_data.index, predictions, color='orange', label='Test Predictions', linestyle='--')
plt.plot(future_dates, future_predictions, color='green', linestyle='--', label='Future Forecast')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.title('Gold Price Forecast')
plt.legend()
plt.grid(True)
plt.show()

# Display future predictions
print("Future Price Predictions:")
print(future_predictions)

```
## OUTPUT:

# ADF-STATISTIC AND P-VALUE
![Screenshot 2024-11-11 182813](https://github.com/user-attachments/assets/c8dc60ff-0c0f-4ff5-bef6-264b0d76c7d3)


### ACF
![Screenshot 2024-11-11 182909](https://github.com/user-attachments/assets/48a95252-fe29-4745-b5dc-b500198ad4d0)

### PACF
![Screenshot 2024-11-11 182925](https://github.com/user-attachments/assets/3f93cfe6-0977-42d6-94cc-7d391e8e5650)


### MSE VALUE
![Screenshot 2024-11-11 182941](https://github.com/user-attachments/assets/759f0325-8215-4ffa-9540-7363642bdf06)


### PREDICTION
![Screenshot 2024-11-11 183017](https://github.com/user-attachments/assets/db109afb-ae2f-41f1-81b7-7f19f79d4f64)



### FINAL PREDICTION
![Screenshot 2024-11-11 183004](https://github.com/user-attachments/assets/2ee14d21-6981-4aa1-8dd0-02c0b434dfdd)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
