# Recurrent Neural Network

# предсказания на больше чем 1 день https://www.udemy.com/deeplearning/learn/v4/questions/3554002

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') # finance.yahoo.com
training_set = dataset_train.iloc[:, 1:2].values
# указываем 1:2 а не просто 1, т.к. нужен массив значений а не просто числа

# Feature Scaling [ Standardisation, Normalisation]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(
  feature_range = (0, 1) # интеравал значений
)
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
# подготавливаем данные для нейронки
for i in range(60, 1258): # с 60 до 1257
  X_train.append(training_set_scaled[i - 60 : i, 0]) # заполняем по 60 элементов
  y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
# переводим все в правильный формат все данные

# Reshaping
# нужно для добавления новых сигналов в модель, погода, цена евро/доллара, ...
X_train = np.reshape(
  X_train,
  ( # keras documentation => Recurrent Layers => input shapes => 3D tensor (array) with shape
    X_train.shape[0], # 1198, # количество строчек # полчаем афтоматически!!!
    X_train.shape[1], # 60 # количество столбцов # полчаем афтоматически!!!
    1 # количество индикаторов, 3 измерение в данных, допустим еще акции apple
  )
)


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential() # regression continuous value много значений предсказываем подряд

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(
  units = 50, # количество элементов(нейронов)
  return_sequences = True, # так как дальше еще слои LSTM, то True
  input_shape = (X_train.shape[1], 1) # структура входного слоя [60, 1]
))
regressor.add(Dropout(
  0.2 # процент засыпания нейронов, убираем переобучение
))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM( units = 50, return_sequences = True
  # input_shape = (X_train.shape[1], 1)) не нужно добавлять, keras сам поймет
))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM( units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM( units = 50)) # default return_sequences = False т.к. конец
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(
  units = 1 # количество нейронов
))

from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Compiling the RNN
regressor.compile(
  optimizer = 'adam', # RMSprop or Adam обычно хорошо для RNN
  loss = root_mean_squared_error # более правильный вариант
  # loss = 'mean_squared_error' # MSE как ошибку ищем? каким методом?
)

# Fitting the RNN to the Training set
regressor.fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 32
)


# Part 3 - Making the predicting and visualising the results
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv') # finance.yahoo.com
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predictied stock price of 2017 #p15l79
dataset_total = pd.concat(
  (dataset_train['Open'], dataset_test['Open']), # берем только Open столбец
  axis = 0 # vertiacal = 0 (по стокам соединяем), horizontal = 1(по столбцам)
)
inputs = dataset_total[
  len(dataset_total) - len(dataset_test) - 60:
].values # подготавливаем данные для нейронки?

inputs = inputs.reshape(-1, 1) # [1, 2, 3] => [[1], [2], [3]] ??? =)
inputs = sc.transform(inputs) # используем предудущую sc функцию/ только scale => transform!
# подготавливаем данные
X_test = []
for i in range(60, 60 + 20): # с 60 до 60 + 20 (количество предсказаний 1 месяц)
  X_test.append(inputs[i - 60 : i, 0]) # заполняем по 60 элементов !!!! 0 !!!!!!
X_test = np.array(X_test)
# переделываем в 3D данные
X_test = np.reshape(
  X_test,
  ( # keras documentation => Recurrent Layers => input shapes => 3D tensor (array) with shape
    X_test.shape[0], # 1198, # количество строчек # полчаем афтоматически!!!
    X_test.shape[1], # 60 # количество столбцов # полчаем афтоматически!!!
    1 # количество индикаторов, 3 измерение в данных, допустим еще акции apple
  )
)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Predicting stock one by one ================================================
input2 = dataset_total[len(dataset_total) - len(dataset_test) - 60: len(dataset_total) - len(dataset_test)].values
input2 = input2.reshape(-1, 1) # [1, 2, 3] => [[1], [2], [3]] ??? =)
input2 = sc.transform(input2) # scale
X_test2 = []
X_test2.append(input2[0:60, 0])
X_test2 = np.array(X_test2)
# переделываем в 3D данные
X_test2 = np.reshape(
  X_test2,
  ( # keras documentation => Recurrent Layers => input shapes => 3D tensor (array) with shape
    X_test2.shape[0], # 1198, # количество строчек # полчаем афтоматически!!!
    X_test2.shape[1], # 60 # количество столбцов # полчаем афтоматически!!!
    1 # количество индикаторов, 3 измерение в данных, допустим еще акции apple
  )
)

predicted_stock_price2 = regressor.predict(X_test2)
for i in range(0, 20):
  # predicted_stock_price2 = sc.inverse_transform(predicted_stock_price2)
  input2 = np.concatenate((input2, [predicted_stock_price2[0]]), axis=0)
  X_test2 = []
  X_test2.append(input2[i:60 + i, 0])
  X_test2 = np.array(X_test2)
  # переделываем в 3D данные
  X_test2 = np.reshape(
    X_test2,
    ( # keras documentation => Recurrent Layers => input shapes => 3D tensor (array) with shape
      X_test2.shape[0], # 1198, # количество строчек # полчаем афтоматически!!!
      X_test2.shape[1], # 60 # количество столбцов # полчаем афтоматически!!!
      1 # количество индикаторов, 3 измерение в данных, допустим еще акции apple
    )
  )
  predicted_stock_price2 = regressor.predict(X_test2)
predicted_stock_price2 = sc.inverse_transform(input2[-20:])
# ============================================================================



# предсказание на 7 дней вперед ----------------------------------------------
days_ahead = 3
# Данные подготавливаем 60 input and 7 outputs
# Creating a data structure with 60 timesteps and 7 output
X_train7 = []
y_train7 = []
# подготавливаем данные для нейронки
for i in range(60, 1258 - days_ahead): # с 60 до 1257
  X_train7.append(training_set_scaled[i - 60 : i, 0]) # заполняем по 60 элементов
  y_train7.append(training_set_scaled[i: i + days_ahead, 0])

X_train7, y_train7 = np.array(X_train7), np.array(y_train7)
# переводим все в правильный формат все данные

# Reshaping
# нужно для добавления новых сигналов в модель, погода, цена евро/доллара, ...
X_train7 = np.reshape(
  X_train7,
  ( # keras documentation => Recurrent Layers => input shapes => 3D tensor (array) with shape
    X_train7.shape[0], # 1198, # количество строчек # полчаем афтоматически!!!
    X_train7.shape[1], # 60 # количество столбцов # полчаем афтоматически!!!
    1 # количество индикаторов, 3 измерение в данных, допустим еще акции apple
  )
)


# Initialising the RNN
regressor7 = Sequential() # regression continuous value много значений предсказываем подряд

# Adding the first LSTM layer and some Dropout regularisation
regressor7.add(LSTM(
  units = 100, # количество элементов(нейронов)
  return_sequences = True, # так как дальше еще слои LSTM, то True
  input_shape = (X_train7.shape[1], 1) # структура входного слоя [60, 1]
))
regressor7.add(Dropout(
  0.2 # процент засыпания нейронов, убираем переобучение
))

# Adding the second LSTM layer and some Dropout regularisation
regressor7.add(LSTM( units = 100, return_sequences = True
  # input_shape = (X_train.shape[1], 1)) не нужно добавлять, keras сам поймет
))
regressor7.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularisation
regressor7.add(LSTM( units = 100, return_sequences = True))
regressor7.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularisation
regressor7.add(LSTM( units = 100, return_sequences = True)) # default return_sequences = False т.к. конец
regressor7.add(Dropout(0.2))

# Adding the fifth LSTM layer and some Dropout regularisation
regressor7.add(LSTM( units = 100)) # default return_sequences = False т.к. конец
regressor7.add(Dropout(0.2))

# Adding the output layer
regressor7.add(Dense(
  units = days_ahead # количество нейронов
))

from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Compiling the RNN
regressor7.compile(
  optimizer = 'adam', # RMSprop or Adam обычно хорошо для RNN
  loss = root_mean_squared_error # более правильный вариант
  # loss = 'mean_squared_error' # MSE как ошибку ищем? каким методом?
)

# Fitting the RNN to the Training set
regressor7.fit(
  X_train7, y_train7,
  epochs = 100,
  batch_size = 32
)

# regressor7 prediction ------
input7 = dataset_total[len(dataset_total) - len(dataset_test) - 60: len(dataset_total) - len(dataset_test)].values
input7 = input7.reshape(-1, 1) # [1, 2, 3] => [[1], [2], [3]] ??? =)
input7 = sc.transform(input7) # scale
X_test7 = []
X_test7.append(input7[0:60, 0])
X_test7 = np.array(X_test7)
# переделываем в 3D данные
X_test7 = np.reshape(
  X_test7,
  ( # keras documentation => Recurrent Layers => input shapes => 3D tensor (array) with shape
    X_test7.shape[0], # 1198, # количество строчек # полчаем афтоматически!!!
    X_test7.shape[1], # 60 # количество столбцов # полчаем афтоматически!!!
    1 # количество индикаторов, 3 измерение в данных, допустим еще акции apple
  )
)

predicted_stock_price7 = regressor7.predict(X_test7)
predicted_stock_price7 = sc.inverse_transform(predicted_stock_price7)


# ----------------------------------------------------------------------------

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Price 20 days')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Price 20 days')
plt.plot(predicted_stock_price2, color = 'green', label = 'Predicted by me 20 days')
plt.plot(predicted_stock_price7.T, color = 'black', label = 'Predicted by me 7 days')
plt.title('Real and Predicted Google Price')
plt.xlabel('Time 2017.1.1 - 2017.1.31')
plt.ylabel('Google Price')
plt.legend()
plt.show()










