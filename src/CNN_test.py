import numpy as np
from skimage import io, color, exposure, transform
from sklearn.cross_validation import train_test_split
import os
import glob
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from matplotlib import pyplot as plt
# from CNN import cnn_model
import pdb
# %matplotlib inline

NUM_CLASSES = 43
IMG_SIZE = 48


def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, IMG_SIZE, IMG_SIZE), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


def leNet():
    model = Sequential()

    model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape = (3, IMG_SIZE, IMG_SIZE)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("sigmoid"))

    model.add(Convolution2D(16, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(120, 1, 1, border_mode='valid'))

    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation("sigmoid"))
    model.add(Dense(43))
    model.add(Activation('softmax'))
    return model




def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


import pandas as pd
test = pd.read_csv('../../GTSRB/test/GT-final_test.csv',sep=';')

X_test = []
y_test = []
i = 0
for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('../../GTSRB/test/Final_Test/Images/',file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)
    
X_test = np.array(X_test)
y_test = np.array(y_test)


# model  = cnn_model()
model = leNet()
model.load_weights('leNet.h5')


y_pred = model.predict_classes(X_test)
print y_test
acc = np.sum(y_pred==y_test)/float(np.size(y_pred))
print("Test accuracy = {}".format(acc))