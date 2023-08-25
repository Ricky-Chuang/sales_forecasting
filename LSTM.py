import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from math import sqrt
import tqdm
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from keras.layers.core import Dense, Activation, Dropout


# Load dataset
df = pd.read_csv('training_data_cleaned.csv')



# Rolling Cross Validation
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



# Normolization
train = df[df['Date'] <= '2012-02-10']
test = df[df['Date'] >= '2012-02-17']

# columns used
variables = ['Weekly_Sales','total_markdown','IsHoliday','Fuel_Price'] 



# Normolization train
scaler = MinMaxScaler()
train_norm_data = pd.DataFrame(scaler.fit_transform(train[variables]), columns=variables, index=train.index)


# Normolization test
scaler = MinMaxScaler()
test_norm_data = pd.DataFrame(scaler.fit_transform(test[variables]), columns=variables, index=test.index)


# Build the structure of X & y
def df_to_X_y(df,window_size):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [a for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)



# Cross Validation to find the best Parameters

# window size
window_size = [1]

# LSTM1
LSTM1 = [32,64]

# batch_size
batch_size = [8,16,32]

# epoch
epoch = [5,10,20]


# Use For Loop to find the best orders with minimun RMSE 
import tensorflow as tf
tss = TimeSeriesSplit(n_splits=4, test_size=12)



fold = 0
model = 0
pred_list = []
rmse_list = []
mse_list = []
mae_list = []
mape_list = []

LSTM1_list = []
LSTM2_list = []
#LSTM3_list = []
epoch_list = []
batch_size_list = []
window_size_list = []
f_list = []



# Use For Loop to find the best orders with minimun RMSE 
import tensorflow as tf
tss = TimeSeriesSplit(n_splits=4, test_size=12)

for train_idx, val_idx in tss.split(train_norm_data):
    train = train_norm_data.iloc[train_idx]
    test = train_norm_data.iloc[val_idx]
    
    seed_value= 42
    tf.compat.v1.set_random_seed(seed_value)

    for ls1 in LSTM1:
        for bs in batch_size:
            for ep in epoch:
                for win in window_size:
                
                    X_train, y_train = df_to_X_y(train, win)
                    X_test, y_test = df_to_X_y(test, win)

                    n_features = 4

                    LSTM_model = Sequential()
                    LSTM_model.add(InputLayer((win,n_features)))
                    LSTM_model.add(LSTM(ls1,return_sequences=False,recurrent_dropout=0.2))
                    LSTM_model.add(Dense(1))
                    LSTM_model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mse','mape'])

                    LSTM_model.fit(X_train, y_train, epochs=ep, batch_size=bs)
        

                    prediction_test = LSTM_model.predict(X_test)
                    prediction_test = pd.DataFrame(prediction_test).rename(columns={0:'Prediction_Sales_test'})

                    prediction_test['1'] = prediction_test['Prediction_Sales_test']
                    prediction_test['2'] = prediction_test['Prediction_Sales_test']
                    prediction_test['3'] = prediction_test['Prediction_Sales_test']

                    temp_test = pd.DataFrame(scaler.inverse_transform(prediction_test),index=df.index[len(train.index)+win:len(train.index)+len(test.index)])
                    temp_test = temp_test[[0]].rename(columns={0:'Prediction_Sales_test'})

                    rmse = np.sqrt(mean_squared_error(df.iloc[len(train.index)+win:len(train.index)+len(test.index)]['Weekly_Sales'], temp_test['Prediction_Sales_test']))
                    rmse_list.append(rmse)

                    mse = mean_squared_error(df.iloc[len(train.index)+win:len(train.index)+len(test.index)]['Weekly_Sales'], temp_test['Prediction_Sales_test'])
                    mse_list.append(mse)

                    mae = mean_absolute_error(df.iloc[len(train.index)+win:len(train.index)+len(test.index)]['Weekly_Sales'], temp_test['Prediction_Sales_test'])
                    mae_list.append(mae)

                    mape = mean_absolute_percentage_error(df.iloc[len(train.index)+win:len(train.index)+len(test.index)]['Weekly_Sales'], temp_test['Prediction_Sales_test'])
                    mape_list.append(mape)

                    LSTM1_list.append(ls1)
                    batch_size_list.append(bs)
                    epoch_list.append(ep)
                    window_size_list.append(win)
                    f_list.append(fold)

                    model += 1
                    print("---------")
                    print("Number of model:", model)
                    print("---------")
    fold += 1                




df_parameter = pd.DataFrame(data={"LSTM1":LSTM1_list,
                                  "epoch":epoch_list,
                                  "batch_size":batch_size_list,
                                  "window_size":window_size_list,
                                  "fold":f_list,
                                  "RMSE":rmse_list,
                                  "MSE":mse_list,
                                  "MAE":mae_list,
                                  "MAPE":mape_list
                                  })


df_parameter_groupby = df_parameter.groupby(['LSTM1','epoch','batch_size','window_size'])['RMSE','MSE','MAE','MAPE'].mean().reset_index()
                                     
min_mean_rmse = df_parameter_groupby['RMSE'].min()
best_rmse = df_parameter_groupby[df_parameter_groupby['RMSE'] == min_mean_rmse]

# best_rmse (30,10,32)
print(best_rmse)



# Build LSTM Model
train = df[df['Date'] <= '2012-02-10']
test = df[df['Date'] >= '2012-02-10']

# columns used
variables = ['Weekly_Sales','total_markdown','IsHoliday','Fuel_Price'] 


# Normolization train
scaler = MinMaxScaler()
train_norm_data = pd.DataFrame(scaler.fit_transform(train[variables]), columns=variables, index=train.index)


# Normolization test
scaler = MinMaxScaler()
test_norm_data = pd.DataFrame(scaler.fit_transform(test[variables]), columns=variables, index=test.index)



window_size = 1

X_train, y_train = df_to_X_y(train_norm_data, window_size)
X_test, y_test = df_to_X_y(test_norm_data, window_size)



import tensorflow as tf
seed_value= 42
tf.compat.v1.set_random_seed(seed_value)

n_features = 4

LSTM_model = Sequential()
LSTM_model.add(InputLayer((window_size,n_features)))
LSTM_model.add(LSTM(32, return_sequences=False,recurrent_dropout=0.2))
LSTM_model.add(Dense(1))
LSTM_model.compile(optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999), loss='mse', metrics=['mse','mape'])


history = LSTM_model.fit(X_train, y_train, 
                         batch_size=32,
                         epochs=10,
                        )


plt.plot(history.history['loss'],label='Traning Loss')
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend()


# Prediction
prediction_test = LSTM_model.predict(X_test)
prediction_test = pd.DataFrame(prediction_test).rename(columns={0:'Prediction_Sales_test'})



# Transfer to actual values
prediction_test['1'] = prediction_test['Prediction_Sales_test']
prediction_test['2'] = prediction_test['Prediction_Sales_test']
prediction_test['3'] = prediction_test['Prediction_Sales_test']

temp_test = pd.DataFrame(scaler.inverse_transform(prediction_test),index=df.index[window_size+105:])
temp_test = temp_test[[0]].rename(columns={0:'Prediction_Sales_test'})


actual_y = df['Weekly_Sales'][df.index[window_size+105:]]
final = pd.concat([actual_y,temp_test],axis=1)


# Plot line chart
fig, ax = plt.subplots(figsize=(20, 6))
half_year_locator = mdates.MonthLocator(interval=2)
year_month_formatter = mdates.DateFormatter("%b - %Y")
ax.xaxis.set_major_locator(half_year_locator)
ax.xaxis.set_major_formatter(year_month_formatter) # formatter for major axis only


ax.plot(df.index[:106],df['Weekly_Sales'][:106], label='Train')
ax.plot(df.index[106:],df['Weekly_Sales'][106:], label='Test')
ax.plot(final.index,final['Prediction_Sales_test'], label='Prediction of LSTM _ test')

plt.title('LSTM Prediction of Weekly Sales', fontsize=20)
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.xticks(rotation=45)
plt.yticks([40000000,50000000,60000000,70000000,80000000],['40M','50M','60M','70M','80M'])
plt.show()



# Results
expected = final['Weekly_Sales']
predictions = final['Prediction_Sales_test']
mse_lstm = mean_squared_error(expected, predictions)
rmse_lstm = sqrt(mse_lstm)
print('LSTM RMSE: %f' % rmse_lstm)

mse_lstm = mean_squared_error(expected, predictions)
print('LSTM MSE: %f' % mse_lstm)

mae_lstm = mean_absolute_error(expected, predictions)
print('LSTM MAE: %f' % mae_lstm)


mape_lstm = round(mean_absolute_percentage_error(expected, predictions),3)
print('LSTM MAPE: %f' % mape_lstm)