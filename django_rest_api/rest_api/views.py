from __future__ import print_function
from rest_framework.views import APIView
from rest_framework.response import Response

from rest_framework import status
from django.core.files.base import ContentFile
from django.http import HttpResponse

import scipy.misc
import PIL
import numpy as np

np.random.seed(1337)  # for reproducibility
from PIL import Image
import PIL.ImageOps
import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import keras
from keras.callbacks import ModelCheckpoint

def index(request):
    return HttpResponse("Hello, world.")


def api_test(request):
    return HttpResponse("One more step :)")


def MODEL(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    img = cv2.resize(img, (56, 56))
    if K.image_data_format() == 'channels_first':
        input_shape = (1, 56, 56)
        img = img.reshape(1, 1, 56, 56)
    else:
        input_shape = (56, 56, 1)
        img = img.reshape(1, 56, 56,1)
   
    
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
    #model.load_weights('/home/codehead/BanglaLekha_Project/django_rest_imageupload_backend/rest_api/weight_current.hdf5')
  # model.load_weights('/root/django_rest_api/rest_api/banglalekhaweights.hdf5')
    model.load_weights('rest_api/weighttraineded.hdf5')
    print("Atkaise ? Abar?")
    img_rows, img_cols = 56, 56

   
   



    value = model.predict_classes(img)
    print(value[0])
    return value[0]
#     return 1;
def get_pred(full_filename):
    img = np.array(Image.open(full_filename).convert('RGB'))
    # img = img.reshape((3,512,512))
    img = scipy.misc.imresize(img, (64,64,3))
    img = np.rollaxis(img, 2, 0)

    # return predict_from_model(img)



class PhotoList(APIView):

    def get(self, request, format=None):
        return Response({'key': 'value'}, status=status.HTTP_201_CREATED)

    def post(self,request,format=None):
        folder = 'uploaded_media/' #request.path.replace("/", "_")
        print("Came Here mate")
        uploaded_filename = request.FILES['file'].name
        BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print("Came Here mate_2")
        # create the folder if it doesn't exist.
        print(BASE_PATH)
        print("no come")
        print(uploaded_filename)
        try:
            os.mkdir(os.path.join(BASE_PATH, folder))
        except:
            pass

        # save the uploaded file inside that folder.
        full_filename = os.path.join(BASE_PATH, folder, uploaded_filename)
        fout = open(full_filename, 'wb+')

        file_content = ContentFile( request.FILES['file'].read() )

        try:
            # Iterate through the chunks.
            for chunk in file_content.chunks():
                fout.write(chunk)
            fout.close()

            image = Image.open(full_filename)

            image = cv2.imread(full_filename)
            image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

            blur = cv2.GaussianBlur(image, (5, 5), 0)
            ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cv2.imwrite(full_filename,th3)

            # image = Image.open(full_filename)
            # if (image.mode != 'P'):
            #     inverted_image = PIL.ImageOps.invert(image)
            #
            #     inverted_image.save(full_filename)

            # plt.imshow(image)
            # plt.show()



            # allpreds = MODEL(full_filename)
            allpreds = MODEL(full_filename)
            # os.remove(full_filename)
           #

            return Response({'key': str(allpreds)}, status=status.HTTP_201_CREATED)
        except Exception as inst:
            raise inst
            return Response({'key': 'NOT SAVED'}, status=status.HTTP_201_CREATED)

        return Response({'key': 'value'}, status=status.HTTP_201_CREATED)



class PhotoDetail(APIView):
    pass
