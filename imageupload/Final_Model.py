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
from PIL import Image
from keras.optimizers import RMSprop

image='/root/projects/django_rest_api/uploaded_media/b_2.jpg'


img = Image.open(image).convert('L')
img = img.resize((56, 56), Image.ANTIALIAS)

img = np.array(img)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(1,56,56)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_class, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#model.load_weights('/home/codehead/Downloads/django_rest_api/imageupload/weights.hdf5')
model.load_weights('/root/projects/django_rest_api/imageupload/weights.hdf5')
print("Atkaise ? Abar?")

value= model.predict_proba(img)
print(value)
value = model.predict_classes(img)
print(value[0])
