import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

start = '2015-01-01'
end = '2022-03-27'

df = data.DataReader('AAPL', 'yahoo', start, end)
# print(df.head())
# df.reset_index(inplace=True)
# df = df.drop([ 'Date','Adj Close'], axis = 1)

# print(df.head())
# plt.plot(df.Close)
# plt.show()
ma100 = df.Close.rolling(100).mean()
# plt.figure(figsize = (12, 6))
# plt.plot(df.Close)
# plt.plot(ma100, 'r')



ma200 = df.Close.rolling(200).mean()
# plt.plot(ma200, 'g')
# plt.show()

#Splitting Data into Traing and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# print(data_testing.shape)
# print(data_training.shape)
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train)


## Load Model

model = load_model('keras_model.h5')


past_100_days = data_training.tail(100) 
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


#Making Prediction

y_predicted = model.predict(x_test)
scaler = scaler.scale_
# print(scaler)
scale_factor = 1/scaler[0]

y_predicted = y_predicted * scale_factor
# print(y_predicted)
y_test = y_test * scale_factor

# plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'b', label = 'Original Price')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
