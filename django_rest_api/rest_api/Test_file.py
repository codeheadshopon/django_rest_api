from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility

import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import keras
from keras.callbacks import ModelCheckpoint

image='/root/projects/django_rest_api/uploaded_media/b_2.jpg'

img = cv2.imread(image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
img = cv2.resize(img, (56, 56))
if K.image_data_format() == 'channels_first':
    input_shape = (1, 56, 56)
    img = img.reshape(1, 1, 56, 56)
else:
    input_shape = (56, 56, 1)
    img = img.reshape(1, 56, 56, 1)
img = img.astype('float32')
img /= 255

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(60, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
print("Atkaise ? ")
# model.load_weights('/home/codehead/BanglaLekha_Project/django_rest_imageupload_backend/rest_api/weight_current.hdf5')
model.load_weights('/root/projects/django_rest_api/rest_api/weight_current.hdf5')
print("Atkaise ? Abar?")
img_rows, img_cols = 56, 56

value = model.predict_classes(img)
print(value[0])
