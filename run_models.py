
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, GRU, Conv2D, MaxPooling2D

# define baseline model
def dffn_model(input_shape, n_outputs, dropout = 0.5, nhl=32):
    # create model
    model = Sequential()
    model.add(Dense(nhl, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define basic lstm model
def lstm_model(input_shape, n_outputs, dropout = 0.5, nhl=32):
    model = Sequential()
    model.add(LSTM(nhl, input_shape=input_shape, dropout=dropout))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define basic gru model
def gru_model(input_shape, n_outputs, dropout = 0.5, nhl=32):
    print("\n\nInside GRU Model")
    print(input_shape)
    model = Sequential()
    model.add(GRU(nhl, input_shape=input_shape, dropout=dropout))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model