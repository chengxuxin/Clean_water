import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.optimizers import Adadelta

train_x = np.load('train_input.npy')
train_y = np.load('train_label.npy')
train_x = train_x.reshape((-1, 30, 40, 1))
train_x = train_x.astype('float32')
train_x /= 255
train_y = to_categorical(train_y, 11)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=[30, 40, 1]))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

batch_size = 16
epochs = 32
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs)

test_x = np.load('test_input.npy')
test_y = np.load('test_label.npy')
test_x = test_x.reshape((-1, 30, 40, 1))
test_x = test_x.astype('float32')
test_x /= 255
test_y = to_categorical(test_y, 11)

loss, accuracy = model.evaluate(test_x, test_y, verbose=1)
print('loss:%.4f accuracy:%.4f' % (loss, accuracy))