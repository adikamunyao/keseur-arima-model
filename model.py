# import relevant libraries
import os
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima


warnings.filterwarnings("ignore")


data_path = os.environ.get("DATA_PATH")

# load data and parse data column
euro = pd.read_csv(
    f"{data_path}/EURO.csv",
    parse_dates=["Date"],
    index_col="Date"
)

euro.head(7)

duplicated_rows = euro.duplicated().sum()
# drop the duplicated rows based on all columns
df = euro.drop_duplicates()
print(f"Dropped {duplicated_rows} duplicate rows")


# create a boxplot for each column
plt.figure(figsize=(10, 8))
sns.boxplot(data=df[['Mean', 'Buy', 'Sell']], palette='Set3')

# set the plot title and y-axis label
plt.title('Boxplot of Exchange Rates')
plt.ylabel('Exchange Rate')

# show the plot
plt.show()


# increase plot size
plt.rcParams["figure.figsize"] = (10, 5)

# plot ACF
plot_acf(df['Mean'], lags=30)

plt.axhline(y=-1.96/np.sqrt(len(df)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df)), linestyle='--', color='gray')
plt.title('Autocorrelation Function (ACF) of Mean')
plt.xlabel('Lags')
plt.ylabel('Correlation')
plt.show()

# Plot PACF
plot_pacf(df['Mean'], lags=30)
plt.axhline(y=-1.96/np.sqrt(len(df)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function (PACF) of Mean')
plt.xlabel('Lags')
plt.ylabel('Correlation')
plt.show()


# >The ACF plot shows a slow decay, and the PACF plot shows a sharp cutoff after lag p, a first-order difference is needed to make the time series stationary. 

# #### 3.3. Stationarity.

# ####  Stationarity Test

# In[8]:


# function to test 
def test_stationarity(series):
    # augmented Dickey-Fuller test
    adf_result = adfuller(series)
    print('Augmented Dickey-Fuller Test:')
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print('\t%s: %.3f' % (key, value))
    if adf_result[0] < adf_result[4]['5%']:
        print('ADF test indicates series is stationary')
    else:
        print('ADF test indicates series is non-stationary')

    # kwiatkowski-Phillips-Schmidt-Shin test
    kpss_result = sm.tsa.stattools.kpss(series)
    print('\nKwiatkowski-Phillips-Schmidt-Shin Test:')
    print('KPSS Statistic: %f' % kpss_result[0])
    print('p-value: %f' % kpss_result[1])
    print('Critical Values:')
    for key, value in kpss_result[3].items():
        print('\t%s: %.3f' % (key, value))
    if kpss_result[0] < kpss_result[3]['5%']:
        print('KPSS test indicates series is stationary')
    else:
        print('KPSS test indicates series is non-stationary')

# call function
test_stationarity(df['Mean'])        


# ## 4. Data PreProcessing

# In[9]:


# drop currency column
final_df = euro.drop("Currency", axis=1)


# #####  Apply Differencing
# The time series is non-stationary, apply differencing to make it stationary.I will apply first-order differencing.

# In[10]:


# apply first order differencing
diff_y = final_df.diff(periods=1).dropna()


# >**After applying differencing, check for stationarity again, and these are results of the Augmented Dickey-Fuller test, the Kwiatkowski-Phillips-Schmidt-Shin test, and the Phillips-Perron test.**

# call function to test stationarity
test_stationarity(diff_y["Mean"])


# ## 5. Determine the Parameters

# Find the optimal order of the ARIMA model
model = auto_arima(diff_y["Mean"], seasonal=False, trace=True,
                   suppress_warnings=True, error_action="ignore")
model

# build the ARIMA model
# AR order
p = None  

# differencing order
d = None  

# MA order
q = None  

kesur_mean = sm.tsa.ARIMA(df['Mean'], order=(2, 0, 3))
kesur_sell = sm.tsa.ARIMA(df['Sell'], order=(2, 0, 3))
kesur_buy = sm.tsa.ARIMA(df['Buy'], order=(1, 0, 3))

# fit the model
mean_results = kesur_mean.fit()
sell_results = kesur_sell.fit()
buy_results = kesur_buy.fit()


euro.head(5)


# forecast for 1 year
forecast_periods = 12
mean_forecast = mean_results.forecast(steps=forecast_periods)
sell_forecast = sell_results.forecast(steps=forecast_periods)
buy_forecast = buy_results.forecast(steps=forecast_periods)

# print the forecasted values
if len(mean_forecast) > 0:
    print("Mean forecast for April 2024:", mean_forecast.mean())
else:
    print("Error: Mean forecast array is empty")
    
if len(sell_forecast) > 0:
    print("Sell forecast for April 2024:", sell_forecast.mean())
else:
    print("Error: Sell forecast array is empty")

if len(buy_forecast) > 0:
    print("Buy forecast for April 2024:", buy_forecast.mean())
else:
    print("Error: Buy forecast array is empty")


# ### Tune for Optimised Model

# # 7. Evaluation


# import neccesary libraries
from scipy import stats

# evaluate the model
residuals = mean_results.resid

# plot histogram of residuals
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=25, density=True, alpha=0.6, color='b')

# plot normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
ax.plot(x, p, 'k', linewidth=2)
ax.set_xlabel('Residuals')
ax.set_ylabel('Density')
ax.set_title('Histogram of Residuals')
plt.show()

# evaluate the model
residuals = sell_results.resid

# plot histogram of residuals
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=25, density=True, alpha=0.6, color='b')

# plot normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
ax.plot(x, p, 'k', linewidth=2)
ax.set_xlabel('Residuals')
ax.set_ylabel('Density')
ax.set_title('Histogram of Residuals')
plt.show()

# evaluate the model
residuals = buy_results.resid

# plot histogram of residuals
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=25, density=True, alpha=0.6, color='b')

# plot normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
ax.plot(x, p, 'k', linewidth=2)
ax.set_xlabel('Residuals')
ax.set_ylabel('Density')
ax.set_title('Histogram of Residuals')
plt.show()


# >residuals are normally distributed

