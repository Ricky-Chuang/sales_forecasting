import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pmdarima as pm
import plotly.graph_objects as go
import statsmodels.tsa.api as smt
from matplotlib import dates
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.model_selection import TimeSeriesSplit


# Load the dataset
df = pd.read_csv('training_data_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])


# Weekly sales data preview
fig, ax = plt.subplots(figsize=(20, 6))
half_year_locator = mdates.MonthLocator(interval=2)
year_month_formatter = mdates.DateFormatter("%b - %Y")
ax.plot(df['Date'],df['Weekly_Sales'])
ax.xaxis.set_major_locator(half_year_locator)
ax.xaxis.set_major_formatter(year_month_formatter) # formatter for major axis only
plt.title('Weekly Sales in Walmart from 2010 to 2012')
plt.ylabel('Weekly Sales')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.yticks([40000000,50000000,60000000,70000000,80000000],['40M','50M','60M','70M','80M'])
plt.show()


# Adfuller Test
def tsplot(df, lags):
    with plt.style.context("bmh"):    
        fig = plt.figure(figsize=(12, 7))
        ts_ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        acf_ax = plt.subplot2grid((2, 2), (1, 0))
        pacf_ax = plt.subplot2grid((2, 2), (1, 1))
        df.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(df)[1]
        ts_ax.set_title('Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(df, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(df, lags=lags, ax=pacf_ax)
        plt.tight_layout()


tsplot(pd.DataFrame(df['Weekly_Sales'].diff().dropna()), 20)



# Rolling Cross Validation
train_data = df[df['Date'] <= '2012-02-10']
test_data = df[df['Date'] >= '2012-02-17']
tss = TimeSeriesSplit(n_splits=4, test_size=12)


# Line chart of cross validation
fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

fold = 0
for train_idx, val_idx in tss.split(train_data):
    train = train_data.iloc[train_idx]
    test = train_data.iloc[val_idx]
    train['Weekly_Sales'].plot(ax=axs[fold],
                          label='Training Set',
                          title=f'Data Train/Test Split Fold {fold}')
    test['Weekly_Sales'].plot(ax=axs[fold],
                         label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1
plt.legend()
plt.show()



# Cross Validation to find the best parameters
p = [0,1,2,3]
q = [0,1,2,3]


# Use For Loop to find the best orders with minimun RMSE 
tss = TimeSeriesSplit(n_splits=4, test_size=12)

fold = 0
pred_list = []
score_list = []
p_list = []
q_list = []
f_list = []
aic_list = []


for train_idx, val_idx in tss.split(train_data):
    train = train_data.iloc[train_idx]
    test = train_data.iloc[val_idx]

    FEATURES = ['total_markdown','IsHoliday','Fuel_Price'] 
    TARGET = 'Weekly_Sales'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]



    for p_order in p:
        for q_order in q:
                arimax_model =  SARIMAX(train['Weekly_Sales'],
                                exog=train[['Fuel_Price','IsHoliday','total_markdown']],
                                order=(p_order,1,q_order),
                                trend='c')
                arimax_model = arimax_model.fit(disp=-1,maxiter=200)
                
                y_pred_arimax = arimax_model.predict(start=X_train.shape[0],
                                    end=X_train.shape[0] + X_test.shape[0]-1,
                                    exog=X_test[['Fuel_Price','IsHoliday','total_markdown']]
                                    )
                
                pred_list.append(y_pred_arimax)
                score = np.sqrt(mean_squared_error(y_test, y_pred_arimax))
                score_list.append(score)
                p_list.append(p_order)
                q_list.append(q_order)
                f_list.append(fold)
                aic_list.append(arimax_model.aic)

    fold += 1


df_parameter = pd.DataFrame(data={"p":p_list,
                                  "q":q_list,
                                  "fold":f_list,
                                  "RMSE":score_list,
                                  "AIC":aic_list})

df_parameter_groupby = df_parameter.groupby(['p','q'])['RMSE','AIC'].mean().reset_index()
                                     
min_mean_rmse = df_parameter_groupby['RMSE'].min()
min_mean_aic = df_parameter_groupby['AIC'].min()
best_rmse = df_parameter_groupby[df_parameter_groupby['RMSE'] == min_mean_rmse]
best_aic = df_parameter_groupby[df_parameter_groupby['AIC'] == min_mean_aic]


# Best AIC (3,1,1)
print(best_aic)



# Build ARIMAX Model
train = df[df['Date'] <= '2012-02-10']
test = df[df['Date'] >= '2012-02-17']

arimax_model =  SARIMAX(train['Weekly_Sales'],
                exog=train[['Fuel_Price','IsHoliday','total_markdown']],
                order=(3,1,1),
                seasonal_order=(0,0,0,0),
                trend='c')
arimax_model = arimax_model.fit(disp=-1,maxiter=200)
arimax_model.summary()



# Predict
y_pred_arimax = arimax_model.predict(start=train.shape[0],
                              end=train.shape[0] + test.shape[0]-1,
                              exog=test[['Fuel_Price','IsHoliday','total_markdown']]
                              )

y_pred_arimax = pd.DataFrame(y_pred_arimax)
y_pred_arimax.reset_index(drop=True, inplace=True)
y_pred_arimax['Date'] = list(test['Date'])
y_pred_arimax['Date'] = pd.to_datetime(y_pred_arimax['Date'] )
y_pred_arimax.rename(columns={'predicted_mean':'Weekly_Sales'},inplace=True)

arimax_model_fittedvalues = pd.DataFrame(arimax_model.fittedvalues)
arimax_model_fittedvalues['Date'] = list(train['Date'])
arimax_model_fittedvalues['Date'] = pd.to_datetime(arimax_model_fittedvalues['Date'])
arimax_model_fittedvalues.rename(columns={0:'Weekly_Sales'},inplace=True)

fig, ax = plt.subplots(figsize=(20, 6))
half_year_locator = mdates.MonthLocator(interval=2)
year_month_formatter = mdates.DateFormatter("%b - %Y")
ax.xaxis.set_major_locator(half_year_locator)
ax.xaxis.set_major_formatter(year_month_formatter) # formatter for major axis only


ax.plot(train['Date'],train['Weekly_Sales'], label='Train')
ax.plot(test['Date'],test['Weekly_Sales'], label='Test')
ax.plot(y_pred_arimax['Date'],y_pred_arimax['Weekly_Sales'], label='Prediction of ARIMAX')

plt.title('ARIMAX Prediction of Weekly Sales', fontsize=20)
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel('Weekly Sales', fontsize=14)
plt.yticks([40000000,50000000,60000000,70000000,80000000],['40M','50M','60M','70M','80M'])
plt.show()



# Result 
expected = test['Weekly_Sales']
predictions = y_pred_arimax['Weekly_Sales']
mse_arimax = mean_squared_error(expected, predictions)
rmse_arimax = sqrt(mse_arimax)
print('ARIMAX RMSE: %f' % rmse_arimax)

mse_arimax = mean_squared_error(expected, predictions)
print('ARIMAX MSE: %f' % mse_arimax)

mae_arimax = mean_absolute_error(expected, predictions)
print('ARIMAX MAE: %f' % mae_arimax)

mape_arimax = round(mean_absolute_percentage_error(expected, predictions),3)
print('ARIMAX MAPE: %f' % mape_arimax)


# Residual Analysis
arimax_model.plot_diagnostics(figsize=(15,12))