# Recurrent Neural Network

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
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
X_train = np.reshape(
  X_train,
  ( # keras documentation => Recurrent Layers => input shapes => 3D tensor (array) with shape
    X_train.shape[0], # 1198, # количество строчек
    X_train.shape[1], # 60 # количество столбцов
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















