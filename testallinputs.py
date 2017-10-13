#! /usr/bin/env python

import keras
import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, GRU, Dropout, Dense, Activation
from run_models import dffn_model, lstm_model, gru_model


def readfile(file):
	return pd.read_csv(file,sep='\t',header=None)

def getxy(data):
	# replace the missing values (currently -1) with nan
    data[data==np.float64(-1)] = np.nan
    data_complete = data[data.notnull().sum(axis=1) >= 15]

    x = data_complete.iloc[:,0:14].values
    assert(sum(sum(x==np.nan))==0)
    print("passed assert 1")

    x = np.reshape(x,(x.shape[0], x.shape[1], 1))

    y = data_complete.iloc[:,14].values
    assert(sum(y==np.nan)==0)
    print("passed assert 2")

    y = np_utils.to_categorical(y)
    return x, y


def getmodel(model_str):
    if model_str == 'gru':
        model = gru_model
    elif model_str == 'lstm':
        model = lstm_model
    elif model_str == 'dffn':
        model = dffn_model
    return model

if __name__ == '__main__':
    
    df = readfile(sys.argv[1])
    numclasses = int(sys.argv[2])
    dimhiddenlayer = int(sys.argv[5])

    X, y = getxy(df)

    model = getmodel(sys.argv[3])
    n_splits = int(sys.argv[4])

    estimator = KerasClassifier(build_fn=model, n_outputs=numclasses, nhl=dimhiddenlayer, input_shape=X.shape[1:])
    kfold = KFold(n_splits=n_splits, shuffle=True)
    results = cross_val_score(estimator, X, y, cv=kfold)
    
    print(results)
    print("\Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    f = open('out.txt', 'a')
    f.write("\nData: %s  Model: %s  Dim. hidden layer: %s n-splits: %i" % (sys.argv[1], sys.argv[3], dimhiddenlayer, n_splits))
    f.write("\nResults: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    f.close()