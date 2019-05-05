#%%:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Reshape,Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
import pandas as pd
import time
from numpy.random import seed
seed(1337)
classes=47
#%%:
train = pd.read_csv('./emnist-balanced-train.csv', header=None)
test = pd.read_csv('./emnist-balanced-test.csv',header=None)
train_data = train.iloc[:, 1:]
train_labels = train.iloc[:, 0]
test_data = test.iloc[:, 1:]
test_labels = test.iloc[:, 0]

y_train=to_categorical(train_labels,classes)
y_test=to_categorical(test_labels,classes)
train_data = train_data.values
test_data = test_data.values
print(y_test)
del train, test,test_labels,train_labels

#%%:
def transpose(data):
   for i in range(len(data+1)):
       flipped = np.fliplr(data[i].reshape(28,28))
       data[i]=np.rot90(flipped).reshape(784,)
   return data
#%%
x_test=transpose(test_data)
x_test = x_test.astype('float32')
x_test /= 255
x_test = np.asarray(x_test)
#%%:
x_train=transpose(train_data)
x_train = x_train.astype('float32')
x_train /= 255
x_train = np.asarray(x_train)
#%%
print("Transposed values")
del test_data,train_data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
#%%:
print(y_train.shape)
epochs=20
batch_size = 64
print("Arranging training and testing data for the Convolutional Neural Network")
# # Arranging training and testing data for the Convolutional Neural Network
model=Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same', activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5,5) , padding = 'same'))
model.add(ELU(alpha=0.1))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5,5) , padding = 'same'))
model.add(ELU(alpha=0.1))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(4,4), padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5,5) , padding = 'same'))
model.add(ELU(alpha=0.1))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(7,7) , padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters=256, kernel_size=(7,7), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(classes, activation='softmax'))

model.summary() 

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=200)
mc = ModelCheckpoint('./hr.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

datagen = ImageDataGenerator(rotation_range=60,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.3,zoom_range=0.2,horizontal_flip=False)
datagen.fit(x_train,augment=True)
train_gen = datagen.flow(x_train, y_train, batch_size=32)
model.fit_generator(train_gen, steps_per_epoch=len(x_train)/20, epochs=epochs,validation_data=(x_test, y_test),callbacks=[es,mc],shuffle=True)
saved_model = tf.keras.models.load_model('./hr.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(x_train, y_train, verbose=0)
_, test_acc = saved_model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
