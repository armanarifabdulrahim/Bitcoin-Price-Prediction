
#%%
#import nessesary module
import numpy as np
import pandas as pd
import os, datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from keras import Sequential, Input
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import TensorBoard, EarlyStopping





#%%
#Step 1) Data Loading
train_csv_fname = 'gemini_BTCUSD_2020_1min_train.csv'
test_csv_fname = 'gemini_BTCUSD_2020_1min_test.csv'

dir_path = os.path.join(os.getcwd(), 'Dataset')

# df = pd.read_csv(os.path.join(dir_path,test_csv_fname), delim_whitespace=True, header=None, names=['Unix Timestamp',	'Date0', 'Date1',
# 	'Symbol',	'Open',	'High',	'Low',	'Close',	'Volume'])

# df['Date'] = df['Date0'] + " " + df['Date1']
# df.drop(columns=['Date0','Date1'])

df = pd.read_csv(os.path.join(dir_path,train_csv_fname))

#%%
#Step 2) Data Inspection
df.info()
df.describe()
df.head()
df.isna().sum()

# plt.figure(figsize=(10,10))
# plt.plot(df['Open'])
# plt.show()



#%%
#Step 3) Data Cleaning
#Convert object to numeric
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df.info()
plt.figure(figsize=(10,10))
plt.plot(df['Open'])
plt.title('Original Train Dataframe')
plt.show()

#Reverse Data 1
open = df['Open']
# open = open[::-1].reset_index(drop=True)
# open.head()
# plt.figure(figsize=(10,10))
# plt.plot(open)
# plt.title('Reversed Train Dataframe')
# plt.show()

#Reverse Data 2
open = open[::-1].values
plt.figure(figsize=(10,10))
plt.plot(open)
plt.title('Reversed Train Dataframe')
plt.show()

#handle NA ()
open = pd.DataFrame(open) #When using Reverse data 2, convert numpy array to dataframe
open = open.interpolate(method='polynomial', order=2)
plt.figure(figsize=(10,10))
plt.plot(open)
plt.title('Interpolated Train Dataframe')
plt.show()

open_train = open.copy()

#%%
#Step 4) Feature Extraction





#%%
#Step 5) Data Pre-processing
# data = open.values
# data = data[::, None]

#Min Max Scaler
minmax = MinMaxScaler() #sparse=False only for ONE HOT ENCODING
data = minmax.fit_transform(open)

window = 100
X_train = []
y_train = []

for i in range(window, len(data)):
    X_train.append(data[i-window:i])
    y_train.append(data[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

#Train test split
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, random_state=123)



#%%
#Step 6) Model Development
model =  Sequential()
model.add(LSTM(64,input_shape=X_train.shape[1:]))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse','mape'])

#Callbacks

logdir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=logdir)
es = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=5, callbacks=[es,tb])



#%%
#Step 7) Model Evaluation

#Load test dataset
#Method 1
# test_df = pd.read_csv(os.path.join(dir_path, train_csv_fname))
test_df = pd.read_csv(os.path.join(dir_path,test_csv_fname), delim_whitespace=True, header=None, names=['Unix Timestamp',	'Date0', 'Date1',
	'Symbol',	'Open',	'High',	'Low',	'Close',	'Volume'])

test_df['Date'] = test_df['Date0'] + " " + test_df['Date1']
test_df.drop(columns=['Date0','Date1'])

#Method 2
#test_df = pd.read_csv(os.path.join(dir_path,test_csv_fname), sep=' ')

#test_df['Open'] = pd.to_numeric(test_df['Open'], errors='coerce') #Open dtype is float
open_test = test_df['Open']
open_test = open_test[::-1].reset_index(drop=True)
open_test = open_test.interpolate(method='polynomial', order=2)
open_test.columns = ['Open']

df_plus = pd.concat((open_test, open_train), axis=0)
df_plus = df_plus[len(df_plus)-window-len(test_df):]


#Transform Test Data
df_plus = minmax.transform(df_plus[::, None])

#Test data until 
df_plus = df_plus[0:len(open_test) + window]

X_test_2 =[]
y_test_2 =[]

for i in range(window,len(df_plus)):
    X_test_2.append(df_plus[i-window:i])
    y_test_2.append(df_plus[i])

X_test_2 = np.array(X_test_2)
y_test_2 = np.array(y_test_2)

#Prediction
predicted_price = model.predict(X_test_2)

#metrics to evaluate the performance
print('Mean Absolute Percentage Error : \n', mean_absolute_percentage_error(y_test_2,predicted_price))


#plot the graph
y_test_2 = minmax.inverse_transform(y_test_2)
predicted_price = minmax.inverse_transform(predicted_price)

plt.figure()
plt.plot(predicted_price,color='green')
plt.plot(y_test_2, color='purple')
plt.legend(['Predicted','Actual'])
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.show()


#metrics to evaluate the performance

print('Mean Absolute Percentage Error : \n', mean_absolute_percentage_error(y_test_2,predicted_price))
print('Mean Absolute Error : \n',mean_absolute_error(y_test_2,predicted_price))





# %%
# Save Model

import pickle
with open('mms.pkl','wb') as f:
    pickle.dump(mms,f)

model.save('bitcoin.h5')
