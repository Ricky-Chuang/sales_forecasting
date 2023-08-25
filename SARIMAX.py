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

# Load dataset
df = pd.read_csv('training_data_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])


# Weekly Sales Data Preview
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

tsplot(pd.DataFrame(df['Weekly_Sales'].diff(52).dropna()), 20)


# Time Series Cross Validation
train_data = df[df['Date'] <= '2012-02-10']
test_data = df[df['Date'] >= '2012-02-17']
tss = TimeSeriesSplit(n_splits=4, test_size=12)

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
P = [0,1,2,3]
Q = [0,1,2,3]

# Use For Loop to find the best orders with minimun RMSE 

tss = TimeSeriesSplit(n_splits=4, test_size=12)

fold = 0
pred_list = []
p_list = []
q_list = []
P_list = []
Q_list = []
f_list = []
aic_list = []

RMSE_list = []
MSE_list = []
MAE_list = []
MAPE_list = []
Fuel_Price_list = []
IsHoliday_list = []
total_markdown_list = []


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
                for P_order in P:
                    for Q_order in Q:
                            sarimax_model = SARIMAX(train['Weekly_Sales'],
                                            exog=train[['Fuel_Price','IsHoliday','total_markdown']],
                                            order=(p_order,0,q_order),
                                            seasonal_order=(P_order,1,Q_order,52),
                                            enforce_invertibility=False,
                                            enforce_stationarity=False,
                                            trend='c').fit(disp=-1,maxiter=200)
                            
                            y_pred_sarimax = sarimax_model.predict(start=X_train.shape[0],
                                                end=X_train.shape[0] + X_test.shape[0]-1,
                                                exog=X_test[['Fuel_Price','IsHoliday','total_markdown']])
                            
                            pred_list.append(y_pred_sarimax)

                            rmse = np.sqrt(mean_squared_error(y_test, y_pred_sarimax))
                            RMSE_list.append(rmse)

                            mse = mean_squared_error(y_test, y_pred_sarimax)
                            MSE_list.append(mse)
                            
                            mae = mean_absolute_error(y_test, y_pred_sarimax)
                            MAE_list.append(mae)
                            
                            mape = round(mean_absolute_percentage_error(y_test, y_pred_sarimax),3)
                            MAPE_list.append(mape)

                            Fuel_Price_list.append(sarimax_model.pvalues[1])
                            IsHoliday_list.append(sarimax_model.pvalues[2])
                            total_markdown_list.append(sarimax_model.pvalues[3])
                            p_list.append(p_order)
                            q_list.append(q_order)
                            P_list.append(P_order)
                            Q_list.append(Q_order)
                            f_list.append(fold)
                            aic_list.append(sarimax_model.aic)

    fold += 1


df_parameter = pd.DataFrame(data={"p":p_list,
                                  "q":q_list,
                                  "Q":Q_list,
                                  "P":P_list,
                                  "fold":f_list,
                                  "RMSE":RMSE_list,
                                  "MSE":MSE_list,
                                  "MAE":MAE_list,
                                  "MAPE":MAPE_list,
                                  "Fuel_Price":Fuel_Price_list,
                                  "IsHoliday":IsHoliday_list,
                                  "total_markdown":total_markdown_list,
                                  "AIC":aic_list})


df_parameter_groupby = df_parameter.groupby(['p','q','P','Q'])['RMSE','MSE','MAE','MAPE','AIC'].mean().reset_index()
df_parameter_groupby_aicover100 = df_parameter_groupby[df_parameter_groupby['AIC']>100]

min_mean_aic = df_parameter_groupby_aicover100['AIC'].min()
best_aic = df_parameter_groupby_aicover100[df_parameter_groupby_aicover100['AIC'] == min_mean_aic]

# best_aic (2,0,3)(0,1,0,52)
print(best_aic)


# Build SARIMAX Model
sarimax_model = SARIMAX(train['Weekly_Sales'],
                exog=train[['Fuel_Price','IsHoliday','total_markdown']],
                order=(2,0,3),
                seasonal_order=(0,1,0,52),
                enforce_invertibility=False,
                trend='c').fit(disp=-1)
sarimax_model.summary()


# Predict
y_pred_sarimax = sarimax_model.predict(start=train.shape[0],
                              end=train.shape[0] + test.shape[0]-1,
                              exog=test[['Fuel_Price','IsHoliday','total_markdown']]
                              )

y_pred_sarimax = pd.DataFrame(y_pred_sarimax)
y_pred_sarimax.reset_index(drop=True, inplace=True)
y_pred_sarimax['Date'] = list(test['Date'])
y_pred_sarimax['Date'] = pd.to_datetime(y_pred_sarimax['Date'] )
y_pred_sarimax.rename(columns={'predicted_mean':'Weekly_Sales'},inplace=True)

sarimax_model_fittedvalues = pd.DataFrame(sarimax_model.fittedvalues)
sarimax_model_fittedvalues['Date'] = list(train['Date'])
sarimax_model_fittedvalues['Date'] = pd.to_datetime(sarimax_model_fittedvalues['Date'])
sarimax_model_fittedvalues.rename(columns={0:'Weekly_Sales'},inplace=True)

fig, ax = plt.subplots(figsize=(20, 6))
half_year_locator = mdates.MonthLocator(interval=2)
year_month_formatter = mdates.DateFormatter("%b - %Y")
ax.xaxis.set_major_locator(half_year_locator)
ax.xaxis.set_major_formatter(year_month_formatter) # formatter for major axis only


ax.plot(train['Date'],train['Weekly_Sales'], label='Train')
ax.plot(test['Date'],test['Weekly_Sales'], label='Test')
ax.plot(y_pred_sarimax['Date'],y_pred_sarimax['Weekly_Sales'], label='Prediction of SARIMAX')



plt.title('SARIMAX Prediction of Weekly Sales', fontsize=20)
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.xticks(rotation=45)
plt.yticks([40000000,50000000,60000000,70000000,80000000],['40M','50M','60M','70M','80M'])
plt.show()


# Result 
expected = test['Weekly_Sales']
predictions = y_pred_sarimax['Weekly_Sales']
mse_sarimax = mean_squared_error(expected, predictions)
rmse_sarimax = sqrt(mse_sarimax)
print('SARIMAX RMSE: %f' % rmse_sarimax)

mse_sarimax = mean_squared_error(expected, predictions)
print('SARIMAX MSE: %f' % mse_sarimax)

mae_sarimax = mean_absolute_error(expected, predictions)
print('SARIMAX MAE: %f' % mae_sarimax)

mape_sarimax = round(mean_absolute_percentage_error(expected, predictions),3)
print('SARIMAX MAPE: %f' % mape_sarimax)


# Residual Analysis
sarimax_model.plot_diagnostics(figsize=(15,12))