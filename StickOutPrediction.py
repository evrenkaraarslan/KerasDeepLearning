import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

import keras

import h5py

import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard

line_count=0

dataset=np.loadtxt(r'C:\Users\mustafa karaarslan\Pictures\newtraining.csv',delimiter=";",dtype=str)

dataset = np.delete(dataset, 0, 1)

print(type(dataset))

map(int,dataset)

print(dataset.dtype)

evren = dataset.astype(np.float)

print("evren data type" ,evren.dtype)

xinput = evren[1:,0:2]

youtput = evren[1:,2]

youtput=np.reshape(youtput, (-1,1))

scaler_xinput = MinMaxScaler()

scaler_youtput = MinMaxScaler()

print(scaler_xinput.fit(xinput))

xinputscale=scaler_xinput.transform(xinput)

print(scaler_youtput.fit(youtput))

youtputscale=scaler_youtput.transform(youtput)

xinput_train, xinput_test, youtput_train, youtput_test = train_test_split(xinputscale, youtputscale)
for row in dataset:

    if line_count == 0:

        print(f'Column names are {", ".join(row)}')

        line_count += 1

    else:

        print(f'\t Current = {row[0]} , Voltage = {row[1]} ,Stick out type = {row[2]}')

        line_count += 1

print(line_count)

keras.callbacks.callbacks.LambdaCallback(on_epoch_begin=None,
                                         on_epoch_end=None,
                                         on_batch_begin=None,
                                         on_batch_end=None,
                                         on_train_begin=None,
                                         on_train_end=None)
model = Sequential()

model.add(Dense(12, input_dim=(2),kernel_initializer='normal',activation='relu'))

model.add(Dense(8, activation='relu'))


model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

history=model.fit(xinput_train, youtput_train , epochs=50, batch_size=32,verbose=1,validation_split=0.2)

model.save("EvrenTrainedData.h5")

print(history.history.keys())

plt.plot(history.history['loss'],color="lime")

plt.plot(history.history['val_loss'],color="yellow")

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('Number of Epoch epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

dataset1=np.loadtxt(r'C:\Users\mustafa karaarslan\Pictures\predictstickout.csv',delimiter=";",dtype=str)

dataset1 = np.delete(dataset1, 0, 1)

print(type(dataset1))

map(int,dataset1)

print(dataset1.dtype)

evren1 = dataset1.astype(np.float)

print("evren data type" ,evren1.dtype)

xnewinput = evren1[1:,0:2]

xnewinput= scaler_xinput.transform(xnewinput)

ynew= model.predict(xnewinput)

ynew = scaler_youtput.inverse_transform(ynew)

xnewinput = scaler_xinput.inverse_transform(xnewinput)

w=0

for i in range(len(xnewinput)):

    print("X=%s, Predicted Stick Out is %s" % (xnewinput[i], ynew[i]))

    w += ynew[i]

w=w/len(xnewinput)

print('average of the stick out is:',w)

for i in range(len(ynew)):

 plt.plot(ynew,color="Aqua")

 plt.title('Predictions')

 plt.ylabel('Stick Out')

 plt.xlabel('Time')

 plt.legend(['Stick Out'], loc='upper left')

plt.show()