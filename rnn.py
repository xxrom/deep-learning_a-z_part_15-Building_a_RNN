# Recurrent Neural Network

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
  X_test.append(inputs[i - 60 : i, 0]) # заполняем по 60 элементов
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


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Price 20 days')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Price 20 days')
plt.title('Real and Predicted Google Price')
plt.xlabel('Time 2017.1.1 - 2017.1.31')
plt.ylabel('Google Price')
plt.legend()
plt.show()










