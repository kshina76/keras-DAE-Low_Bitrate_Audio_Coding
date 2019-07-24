import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def Denoising_Auto_Encoder():
    inputs = Input(shape=(513,))
    x = Dense(1024,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    outputs = Dense(513)(x)
    model = Model(input=inputs, output=outpus)
    model.summary()
    
    return model


if __name__ == '__main__':
    model = Denoising_Auto_Encoder()
    adam = Adam(lr=0.0002)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])