from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility

import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import cPickle,gzip,sys
import keras
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras.optimizers import RMSprop

nb_class=60
img_rows,img_cols=56,56
def dataset_load(path):
    if path.endswith(".gz"):
        f=gzip.open(path,'rb')
    else:
        f=open(path,'rb')

    if sys.version_info<(3,):
        data=cPickle.load(f)
    else:
        data=cPickle.load(f,encoding="bytes")
    f.close()
    return data
#image='/root/projects/django_rest_api/uploaded_media/b_2.jpg'
(x_train,y_train),(x_test,y_test) = dataset_load('./BanglaLekha_Basic_Numerals.pkl.gz')
	
	
if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
test_images=[]
test_labels=[]



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


filepath="weights-improvement-{epoch:02d}-{val_acc:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.load_weights('weighttrained.hdf5')
model.fit(x_train, y_train, batch_size=128
, epochs=120, verbose=1, validation_data=(x_test, y_test),callbacks=callbacks_list)
print("Atkaise ? Abar?")

