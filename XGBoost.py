import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import TimeSeriesSplit


# Load dataset
df = pd.read_csv('training_data_cleaned.csv')


# Create feature
df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime
df['week'] =df['Date'].dt.week
df['month'] =df['Date'].dt.month 
df['year'] =df['Date'].dt.year
df['quarter'] = df['Date'].dt.quarter
df = df.set_index(df['Date'])


# Lag Feature
def add_lags(df):
    target_map = df['Weekly_Sales'].to_dict()
    df['lag364'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    return df

df = add_lags(df)


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
# Number of n_estimators in XGBoost
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 9)]

# Maximum number of levels
max_depth = [3,4,5,6]

# # Learning rate
learning_rate = [0.01, 0.1, 0.2]


tss = TimeSeriesSplit(n_splits=4, test_size=12)

fold = 0
preds = []
RMSE_list = []
MSE_list = []
MAE_list = []
MAPE_list = []
number_of_estimators = []
maximun_depth = []
l_rate = []
f = []

for train_idx, val_idx in tss.split(train_data):
    train = train_data.iloc[train_idx]
    test = train_data.iloc[val_idx]

    FEATURES = ['total_markdown','IsHoliday','Fuel_Price','lag364']
    TARGET = 'Weekly_Sales'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]



  #  reg = XGBRegressor()
    for n in n_estimators:
        for md in max_depth:
           for lr in learning_rate:
              reg = XGBRegressor(booster='gbtree',    
                                n_estimators=n,
                                objective='reg:linear',
                                max_depth=md,
                                learning_rate=lr,
                                seed=42,
                                )
              reg.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      verbose=100)


              y_pred = reg.predict(X_test)
              preds.append(y_pred)

              rmse = np.sqrt(mean_squared_error(y_test, y_pred))
              RMSE_list.append(rmse)
              
              mse = mean_squared_error(y_test, y_pred)
              MSE_list.append(mse)
              
              mae = mean_absolute_error(y_test, y_pred)
              MAE_list.append(mae)
              
              mape = round(mean_absolute_percentage_error(y_test, y_pred),3)
              MAPE_list.append(mape)       

              number_of_estimators.append(n)
              maximun_depth.append(md)
              l_rate.append(lr)
              f.append(fold)
    fold += 1



df_parameter = pd.DataFrame(data={"n_estimators":number_of_estimators,
                                  "max_depth":maximun_depth,
                                  "learning_rate":l_rate,
                                  "fold":f,
                                  "RMSE":RMSE_list,
                                  "MSE":MSE_list,
                                  "MAE":MAE_list,
                                  "MAPE":MAPE_list,
                                  })

df_parameter_groupby = df_parameter.groupby(['n_estimators',
                                     'max_depth',
                                     'learning_rate'
                                     ])['RMSE','MSE','MAE','MAPE'].mean().reset_index()
                                     

min_mean_rmse = df_parameter_groupby['RMSE'].min()
best_rmse = df_parameter_groupby[df_parameter_groupby['RMSE'] == min_mean_rmse]


# best_rmse (100, 3, 0.1)
print(best_rmse)


# Build XGBoost Model
target = ["Weekly_Sales"]
used_cols = ['total_markdown','IsHoliday','Fuel_Price','lag364'] 

X_train = train_data[used_cols]
X_test = test_data[used_cols]
y_train = train_data[target]
y_test = test_data[target]


xgb_model = XGBRegressor(booster='gbtree',    
                         n_estimators=100,
                         objective='reg:linear',
                         max_depth=3,
                         learning_rate=0.1,
                         seed=42
                         )

# Model fit
xgb_model.fit(X_train, y_train)
            
# Model predict
y_pred_xgb = xgb_model.predict(X_test)

# RMSE
rmse_xgb = np.sqrt(np.mean((y_test['Weekly_Sales'] - y_pred_xgb)**2))
print('XGboost RMSE:',rmse_xgb)


# Predict
y_pred_xgb = pd.DataFrame(y_pred_xgb)
y_pred_xgb.reset_index(drop=True, inplace=True)
y_pred_xgb['Date'] = list(test_data['Date'])
y_pred_xgb['Date'] = pd.to_datetime(y_pred_xgb['Date'] )
y_pred_xgb.rename(columns={0:'Weekly_Sales'},inplace=True)


fig, ax = plt.subplots(figsize=(20, 6))
half_year_locator = mdates.MonthLocator(interval=2)
year_month_formatter = mdates.DateFormatter("%b - %Y")
ax.xaxis.set_major_locator(half_year_locator)
ax.xaxis.set_major_formatter(year_month_formatter) 


ax.plot(train_data['Date'],y_train['Weekly_Sales'], label='Train')
ax.plot(test_data['Date'],y_test['Weekly_Sales'], label='Test')
ax.plot(y_pred_xgb['Date'],y_pred_xgb['Weekly_Sales'], label='Prediction of XGBoost')


plt.title('XGBoost Prediction of Weekly Sales', fontsize=20)
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.xticks(rotation=45)
plt.yticks([40000000,50000000,60000000,70000000,80000000],['40M','50M','60M','70M','80M'])
plt.show()

print('XGBoost RMSE:', rmse_xgb)


# Result 
expected = y_test['Weekly_Sales']
predictions_xgb = y_pred_xgb['Weekly_Sales']
mse_xgb = mean_squared_error(expected, predictions_xgb)
rmse_xgb = sqrt(mse_xgb)
print('XGBoost RMSE: %f' % rmse_xgb)

mse_xgb = mean_squared_error(expected, predictions_xgb)
print('XGBoost MSE: %f' % mse_xgb)


mae_xgb = mean_absolute_error(expected, predictions_xgb)
print('XGBoost MAE: %f' % mae_xgb)


mape_xgb = round(mean_absolute_percentage_error(expected, predictions_xgb),3)
print('XGBoost MAPE: %f' % mape_xgb)


# Feature importance
def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'Feature Importance',fontsize=20,pad=20)
    plt.xlabel('Feature Importance',fontsize=16,labelpad=25)
    plt.ylabel('Feature Names',fontsize=16,labelpad=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


plot_feature_importance(xgb_model.feature_importances_,used_cols,'XGBoost ')