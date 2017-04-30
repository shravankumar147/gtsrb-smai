import numpy as np
from skimage import io, color, exposure, transform
from sklearn.cross_validation import train_test_split
import os
import glob
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Merge
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from matplotlib import pyplot as plt
# %matplotlib inline

NUM_CLASSES = 43
IMG_SIZE = 48


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


def get_class(img_path):
    return int(img_path.split('/')[-2])


# print("Error in reading X.h5. Processing all images...")
root_dir = '../../GTSRB/train/Final_Training/Images/'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
c = 0 
for img_path in all_img_paths:
    try:
        if c >1000:
            break
        c += 1
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

        if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
    except (IOError, OSError):
        print('missed', img_path)
        pass

X = np.array(imgs, dtype='float32')
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

with h5py.File('X.h5','w') as hf:
    hf.create_dataset('imgs', data=X)
    hf.create_dataset('labels', data=Y)



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

model = leNet()
# let's train the model using SGD + momentum (how original).
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))




batch_size = 32
nb_epoch = 30

model.fit(X, Y,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.2,
          shuffle=True,
          callbacks=[LearningRateScheduler(lr_schedule),
                    ModelCheckpoint('leNet.h5',save_best_only=True)]
            )

