import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tools.eval_measures import rmse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# Load data
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
df = df.fillna(0)


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

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 9)]


# Number of features to consider at every split
max_features = ['auto', 'sqrt', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]




tss = TimeSeriesSplit(n_splits=4, test_size=12)

fold = 0
preds = []
number_of_estimators = []
maximun_feature = []
RMSE_list = []
MSE_list = []
MAE_list = []
MAPE_list = []
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



    for n in n_estimators:
        for mf in max_features:
            rf_model = RandomForestRegressor(n_estimators=n, 
                                n_jobs=-1, 
                                max_features=mf,
                                bootstrap=True,
                                oob_score=True,
                                random_state=42)
            
            rf_model.fit(X_train, y_train)

            y_pred = rf_model.predict(X_test)
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
            maximun_feature.append(mf)
            f.append(fold)
    fold += 1



df_parameter = pd.DataFrame(data={"n_estimators":number_of_estimators,
                                  "max_features":maximun_feature,
                                  "fold":f,
                                  "RMSE":RMSE_list,
                                  "MSE":MSE_list,
                                  "MAE":MAE_list,
                                  "MAPE":MAPE_list,
                                  })

df_parameter_groupby = df_parameter.groupby(['n_estimators',
                                     'max_features',
                                     ])['RMSE','MSE','MAE','MAPE'].mean().reset_index()

min_mean_rmse = df_parameter_groupby['RMSE'].min()
best_rmse = df_parameter_groupby[df_parameter_groupby['RMSE'] == min_mean_rmse]


# best_rmse (300, 0.7)
print(best_rmse)



# Build Random Forest Model
target = ["Weekly_Sales"]
used_cols = ['total_markdown','IsHoliday','Fuel_Price','lag364'] 

X_train = train_data[used_cols]
X_test = test_data[used_cols]
y_train = train_data[target]
y_test = test_data[target]


rf_model = RandomForestRegressor(n_estimators=300, 
                                 n_jobs=-1, 
                                 max_features=0.7,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=42)

# Model fit
rf_model.fit(X_train, y_train)
            
# Model predict
y_pred_rf = rf_model.predict(X_test)

# RMSE
rmse_rf = np.sqrt(np.mean((y_test['Weekly_Sales'] - y_pred_rf)**2))
print('Random Forest RMSE:',rmse_rf)


# Plot line chart
y_pred_rf = pd.DataFrame(y_pred_rf)
y_pred_rf.reset_index(drop=True, inplace=True)
y_pred_rf['Date'] = list(test_data['Date'])
y_pred_rf['Date'] = pd.to_datetime(y_pred_rf['Date'] )
y_pred_rf.rename(columns={0:'Weekly_Sales'},inplace=True)


fig, ax = plt.subplots(figsize=(20, 6))
half_year_locator = mdates.MonthLocator(interval=2)
year_month_formatter = mdates.DateFormatter("%b - %Y")
ax.xaxis.set_major_locator(half_year_locator)
ax.xaxis.set_major_formatter(year_month_formatter) 


ax.plot(train_data['Date'],y_train['Weekly_Sales'], label='Train')
ax.plot(test_data['Date'],y_test['Weekly_Sales'], label='Test')
ax.plot(y_pred_rf['Date'],y_pred_rf['Weekly_Sales'], label='Prediction of Random Forest')


plt.title('Random Forest Prediction of Weekly Sales', fontsize=20)
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.xticks(rotation=45)
plt.yticks([40000000,50000000,60000000,70000000,80000000],['40M','50M','60M','70M','80M'])
plt.show()

print('Random Forest RMSE:', rmse_rf)


# Result 
expected = y_test['Weekly_Sales']
predictions_rf = y_pred_rf.set_index(y_pred_rf['Date'],drop=True)[['Weekly_Sales']]
mse_rf = mean_squared_error(expected, predictions_rf)
rmse_rf = sqrt(mse_rf)
print('Random Forest RMSE: %f' % rmse_rf)

mse_rf = mean_squared_error(expected, predictions_rf)
print('Random Forest MSE: %f' % mse_rf)

mae_rf = mean_absolute_error(expected, predictions_rf)
print('Random Forest MAE: %f' % mae_rf)

mape_rf = round(mean_absolute_percentage_error(expected, predictions_rf),3)
print('Random Forest MAPE: %f' % mape_rf)


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