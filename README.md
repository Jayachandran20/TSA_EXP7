## DEVELOPED BY: S JAIGANESH
## REGISTER NO: 212222240037
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file
data = pd.read_csv('raw_sales.csv', index_col=0, parse_dates=True)

# Display the first few rows (GIVEN DATA)
print("GIVEN DATA:")
print(data.head())

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(data['price'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit an AutoRegressive (AR) model with 13 lags
model = AutoReg(train['price'], lags=13)
model_fit = model.fit()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test['price'], predictions)
print('Mean Squared Error:', mse)

# Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plt.figure(figsize=(10,6))
plt.subplot(211)
plot_pacf(train['price'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['price'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

# PREDICTION
print("PREDICTION:")
print(predictions)

# Plot the test data and predictions (FINAL PREDICTION)
plt.figure(figsize=(10,6))
plt.plot(test.index, test['price'], label='Actual Price')
plt.plot(test.index, predictions, color='red', label='Predicted Price')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
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
![Screenshot 2024-11-11 183004](https://github.com/user-attachments/assets/60a61a4b-4120-4150-afa3-dc3022bfac0c)


### FINAL PREDICTION
![Screenshot 2024-11-11 183017](https://github.com/user-attachments/assets/d0e52042-fa42-44da-b162-6f56dfdc9389)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
