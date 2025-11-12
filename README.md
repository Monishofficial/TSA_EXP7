# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 12-11-2025



### AIM:
To Implementat an Auto Regressive Model using Python

### ALGORITHM:

1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.

### PROGRAM:

Developed By: MONISH N
Reg No: 212223240097
```

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
df = pd.read_csv("/content/airlines_flights_data.csv")
print("Columns in df:", df.columns)
print("First 5 rows of df:\n", df.head())
df.rename(columns={'15': 'days_left', '58315': 'price'}, inplace=True)
price_over_days_left = df.groupby('days_left')['price'].mean()
price_over_days_left = price_over_days_left.sort_index()
print("\nFirst 5 entries of the new time series 'price_over_days_left':")
print(price_over_days_left.head())
result = adfuller(price_over_days_left.dropna())
print('\nADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

train_size = int(len(price_over_days_left) * 0.8)
train, test = price_over_days_left.iloc[:train_size], price_over_days_left.iloc[train_size:]

# Adjust lags for ACF and PACF plots to be less than 50% of the sample size
max_lags = int(len(price_over_days_left) / 2) - 1 # Ensure lags are less than N/2
plot_acf(price_over_days_left.dropna(), lags=max_lags)
plot_pacf(price_over_days_left.dropna(), lags=max_lags)
plt.show()

# Check if train has enough observations for lags=5
if len(train) > 5:
    model = AutoReg(train, lags=5).fit()
    print(model.summary())

    preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    
    # Assign the test set index to the predictions for correct plotting and alignment
    preds.index = test.index

    error = mean_squared_error(test, preds)
    print("Mean Squared Error:", error)

    plt.figure(figsize=(10,5))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, preds, label='Predicted', color='red')
    plt.legend()
    plt.title("Price over Days Left: Actual vs Predicted")
    plt.show()
else:
    print("Not enough data in the training set to fit AutoReg with lags=5.")

```
### OUTPUT:

GIVEN DATA

<img width="777" height="427" alt="image" src="https://github.com/user-attachments/assets/88057b28-0942-4a73-bde3-6900f8e6ba59" />

PACF - ACF


<img width="565" height="432" alt="image" src="https://github.com/user-attachments/assets/8ea4f1a8-85bd-43f7-ba54-0492acac8536" />


PREDICTION

<img width="292" height="33" alt="image" src="https://github.com/user-attachments/assets/f3de7daa-97e4-4eb0-954f-b35ec99bf03e" />


FINIAL PREDICTION

<img width="861" height="448" alt="image" src="https://github.com/user-attachments/assets/5f9f0bf1-1c86-4e4b-8644-83972f42051f" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.




