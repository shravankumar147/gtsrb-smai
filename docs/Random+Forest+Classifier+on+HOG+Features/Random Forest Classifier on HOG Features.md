
# The German Traffic Sign Recognition Benchmark


```python
# importing the required packages

import numpy as np
import matplotlib.pyplot as plt
import os, glob
import pickle
import pandas as pd
import csv
import cv2
import imutils
from PIL import Image
from skimage import exposure, feature, transform
%matplotlib inline

# classification required packages

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline
```

## Helper Functions


```python
# helper functions 

# function for reading the images
# arguments: path to the traffic sign data, for example '../../GTSRB/train/Final_Training/Images/'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example '../dataset/GTSRB/train/Final_Training/Images/'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def get_csv(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.csv')]


def showimg_n_hog(grayimg,hogImage):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(grayimg)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    ax2.axis('off')
    ax2.imshow(hogImage, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
    
# Functions for testimages    
testpath="../dataset/GTSRB/test/Final_Test/Images/"
def loadtestimages_from_path(testpath):
    print("[INFO] reading all test images from directory\n")
    gtFile = get_csv(testpath)
    filename = gtFile[0]
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=';', quoting=csv.QUOTE_NONE)
    reader.next()
    testfiles = list(reader)
    timg = []
    testimg = []
#     tlbl = []
    for i in testfiles:
    #     print (i[0],i[-1])
        fname = os.path.join(testpath,i[0])
        timg.append(fname)
        tim = plt.imread(fname)
        testimg.append(tim)
#         label = i[-1]
#         tlbl.append(label)
    np.save("Image_n_Labels/testimagenames.npy",timg)
    np.save("Image_n_Labels/testimages.npy",testimg)
#     np.save("Image_n_Labels/testimagelabels.npy",tlbl)
    return timg,testimg
    
def loadtestimages_from_npy():
    print("[INFO] loading from .npy\n")
    timg = np.load("Image_n_Labels/testimagenames.npy")
    testimg = np.load("Image_n_Labels/testimages.npy")
    print("[INFO] DONE!loaded from .npy\n")
#     tlbl = np.load("Image_n_Labels/testimagelabels.npy")
    return timg,testimg
```

## load training images


```python
if os.path.isfile("Image_n_Labels/trainImages.npy") &  os.path.isfile("Image_n_Labels/trainLabels.npy") :
    X = np.load("Image_n_Labels/trainImages.npy")
    y = np.load("Image_n_Labels/trainLabels.npy")
    print("[INFO] Training images and labels are loaded in variables ==> X,y")
    print("[INFO] Number of training Images {} \nNumber of Labels {}".format(len(X), len(y)))
else:    
    # training images and labels
    trainImages, trainLabels = readTrafficSigns('../dataset/GTSRB/train/Final_Training/Images/')
    np.save("Image_n_Labels/trainImages.npy",trainImages)
    np.save("Image_n_Labels/trainLabels.npy",trainLabels)
    print("[INFO] training images and labels are read from the dataset directory")
    print("[INFO] training images saved to Image_n_Labels/trainingImages.npy for further use")
    print("[INFO] training labels saved to Image_n_Labels/trainingLabels.npy for further use")
```

    [INFO] Training images and labels are loaded in variables ==> X,y
    [INFO] Number of training Images 39209 
    Number of Labels 39209


## load test images


```python
if (os.path.isfile("Image_n_Labels/testimagenames.npy") &  os.path.isfile("Image_n_Labels/testimages.npy")):
    timg,testimg = loadtestimages_from_npy()
else:
    timg,testimg = loadtestimages_from_path(testpath)
```

    [INFO] loading from .npy
    
    [INFO] DONE!loaded from .npy
    



```python
print("[INFO] There are total {} test images availabe".format(len(timg)))
print("[INFO] Example {}".format(timg[12629]))
plt.imshow(plt.imread(timg[0]))
```

    [INFO] There are total 12630 test images availabe
    [INFO] Example ../dataset/GTSRB/test/Final_Test/Images/12629.ppm





    <matplotlib.image.AxesImage at 0x7fcb466331d0>




![png](output_8_2.png)



```python
y.astype("float")
```




    array([  0.,   0.,   0., ...,  42.,  42.,  42.])




```python
# simple test on single image for HoG features
n = np.random.randint(0,len(X))
i1 = X[n]
grayim = rgb2gray(i1)
gI1 = transform.resize(grayim,(40,40))
# gI2 = cv2.resize(grayim, (40, 40), interpolation = cv2.INTER_CUBIC)
(H, hogImage) = feature.hog(gI1, orientations=9, pixels_per_cell=(8,8),
    cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255)).astype("uint8")

showimg_n_hog(gI1, hogImage)
print(len(H))
```


![png](output_10_0.png)


    576


## Extract HoG features over all training images 


```python
if os.path.isfile("HoGFeatures/HoGfeatures.npy") & os.path.isfile("HoGFeatures/HoGvisualize.npy") :
    print("[INFO] loading from file ... ")
    hogfeat = np.load("HoGFeatures/HoGfeatures.npy")
    hogviz = np.load("HoGFeatures/HoGvisualize.npy")
    
    print("HoG features are loaded from HoGfeatures.npy to variable ==> hogfeat")
    print("HoG visualizations are loaded from HoGvisualize.npy to variable ==> hogviz")
else:
    print("[INFO] HoGfeatures.npy does not found")
    Hviz = []
    Hfeat = []
    for i in range(0,len(X)):
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(X)))
        I = X[i]
        grayim = rgb2gray(I)
        grayim = transform.resize(grayim,(40,40))

        (H_4x4, hogImage) = feature.hog(grayim, orientations=9, pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255)).astype("uint8")
#         hogImage = hogImage.astype("uint8")
        Hviz.append(hogImage)
        Hfeat.append(H_4x4)
        # save the features using numpy save with .npy extention 
        # which reduced the storage space by 4times compared to pickle
    np.save("HoGFeatures/HoGfeatures.npy", Hfeat)
    np.save("HoGFeatures/HoGvisualize.npy", Hviz)
    print("[INFO] HoGfeatures.npy are saved")  
    print("[INFO] HoGvisualize.npy are saved")
```

    [INFO] loading from file ... 
    HoG features are loaded from HoGfeatures.npy to variable ==> hogfeat
    HoG visualizations are loaded from HoGvisualize.npy to variable ==> hogviz


## Extract HoG features over all testing images 


```python
if os.path.isfile("HoGFeatures/HoGfeatures_test.npy") & os.path.isfile("HoGFeatures/HoGvisualize_test.npy") :
    hogfeat_test = np.load("HoGFeatures/HoGfeatures_test.npy")
    hogviz_test = np.load("HoGFeatures/HoGvisualize_test.npy")
    
    print("HoG features are loaded from HoGfeatures_test.npy to variable ==> hogfeat_test")
    print("HoG visualizations are loaded from HoGvisualize_test.npy to variable ==> hogviz_test")
else:
    print("HoGfeatures_test.npy does not found")
    Hviz = []
    Hfeat = []
    for i in range(0,len(X)):
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(X)))
        I = X[i]
        grayim = rgb2gray(I)
        grayim = transform.resize(grayim,(40,40))

        (H_4x4, hogImage) = feature.hog(grayim, orientations=9, pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255)).astype("uint8")
    #         hogImage = hogImage.astype("uint8")
        Hviz.append(hogImage)
        Hfeat.append(H_4x4)
        # save the features using numpy save with .npy extention 
        # which reduced the storage space by 4times compared to pickle
    np.save("HoGFeatures/HoGfeatures_test.npy", Hfeat)
    np.save("HoGFeatures/HoGvisualize_test.npy", Hviz)
    print("HoGfeatures_test.npy are saved")  
    print("HoGvisualize_test.npy are saved")
```

    HoG features are loaded from HoGfeatures_test.npy to variable ==> hogfeat_test
    HoG visualizations are loaded from HoGvisualize_test.npy to variable ==> hogviz_test


## converting datatype to float: training data


```python
Xhog = np.array(hogfeat).astype("float")
y = y.astype("float")
print(Xhog.shape,y.shape)
print(np.unique(y))
```

    ((39209, 2916), (39209,))
    [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
      15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.  28.  29.
      30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.  41.  42.]


## converting datatype to float: test data


```python
X_t = np.array(hogfeat_test).astype("float")
print(X_t.shape)
```

    (12630, 2916)



```python
for i in range(0,len(np.unique(y))):
    yi = np.where(y==i)
#     print("label: {}  \n indices: {} ".format(i,yi))
```


```python
features = Xhog
labels = y
Xtest = X_t

# take the  data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(features,
    labels, test_size=0.25, random_state=42)
 
# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
    test_size=0.1, random_state=84)
 
# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))
```

    training data points: 26465
    validation data points: 2941
    testing data points: 9803


## Random Forest Classifier


```python
n_features = Xhog.shape[1]
print(np.sqrt(n_features))
```

    54.0



```python
# MLP
if os.path.isfile("clf/clf_rf_hog.pkl"):
    print("[INFO] loading classifier: Random Forest trained on HoG features...")
    svc = joblib.load("clf/clf_rf_hog.pkl")
    print("[INFO] Classifer is loaded as instance ::svc::")
else:
    print("[INFO] pre-trained classifier not found. \n Training Classifier Random Forest")
    rf = RandomForestClassifier()
    rf.fit(trainData,trainLabels)
#     scores = cross_val_score(rf,testData,testLabels)
    rf.fit(trainData,trainLabels)
    print("[INFO] Succefully trained the classsifier. \n Saving the classifier for further use")
    joblib.dump(rf, 'clf/clf_rf_hog.pkl') 
    print("[INFO] Classifier Saved")

```

    [INFO] pre-trained classifier not found. 
     Training Classifier Random Forest
    [INFO] Succefully trained the classsifier. 
     Saving the classifier for further use
    [INFO] Classifier Saved



```python
print("accuracy on training data: {}".format(rf.score(trainData,trainLabels)))
```

    accuracy on training data: 0.999319856414



```python
print("accuracy on test data: {}".format(rf.score(testData,testLabels)))
```

    accuracy on test data: 0.881056819341



```python
print("accuracy on validation data: {}".format(rf.score(valData,valLabels)))
```

    accuracy on validation data: 0.891193471608


### cross validation accuracy


```python
cv_score = cross_val_score(rf,testData,testLabels,cv=5)
print("mean cross-validation score: {}".format(np.mean(cv_score)))
```

    mean cross-validation score: 0.830144524554



```python
rf.predict(testData[8000])
```




    array([ 25.])




```python
y_pred = rf.predict(Xtest)
```


```python
y_pred
```




    array([ 16.,   1.,  38., ...,   1.,   7.,  10.])




```python
import csv
with open('submission_rf.csv', "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for line in zip(timg,y_pred):
            writer.writerow(line)
```

### final classification report


```python
predictions = rf.predict(testData)
 
# show a final classification report demonstrating the accuracy of the classifier
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))
```

    EVALUATION ON TESTING DATA
                 precision    recall  f1-score   support
    
            0.0       0.79      0.75      0.77        44
            1.0       0.76      0.85      0.80       594
            2.0       0.68      0.82      0.75       560
            3.0       0.80      0.78      0.79       348
            4.0       0.88      0.90      0.89       529
            5.0       0.68      0.64      0.66       462
            6.0       0.90      0.90      0.90        81
            7.0       0.77      0.83      0.80       336
            8.0       0.82      0.73      0.77       385
            9.0       0.94      0.96      0.95       359
           10.0       0.95      0.96      0.95       461
           11.0       0.79      0.92      0.85       291
           12.0       0.97      0.99      0.98       548
           13.0       0.99      1.00      1.00       549
           14.0       0.99      0.86      0.92       197
           15.0       0.98      0.90      0.94       152
           16.0       0.99      0.94      0.96       108
           17.0       0.99      0.99      0.99       286
           18.0       0.92      0.97      0.94       288
           19.0       0.85      0.85      0.85        53
           20.0       0.87      0.79      0.83        90
           21.0       0.94      0.82      0.88        82
           22.0       0.96      0.98      0.97       107
           23.0       0.87      0.88      0.88       138
           24.0       0.98      0.84      0.90        68
           25.0       0.88      0.96      0.92       369
           26.0       0.87      0.86      0.87       152
           27.0       0.90      0.78      0.84        60
           28.0       0.81      0.74      0.77       129
           29.0       0.87      0.44      0.59        77
           30.0       0.91      0.62      0.74       118
           31.0       0.96      0.95      0.96       201
           32.0       0.96      0.94      0.95        48
           33.0       0.96      0.97      0.96       182
           34.0       1.00      0.85      0.92       117
           35.0       0.99      0.97      0.98       290
           36.0       0.99      0.90      0.95       105
           37.0       0.96      0.90      0.93        60
           38.0       0.98      0.98      0.98       517
           39.0       1.00      0.90      0.95        69
           40.0       0.94      0.58      0.72        77
           41.0       1.00      0.78      0.88        60
           42.0       0.92      0.80      0.86        56
    
    avg / total       0.89      0.88      0.88      9803
    


### Testing on random images


```python
def testing_on_data():
    n = np.random.randint(0, high=len(testLabels))
    test1 =X[n]
    t1_true = y[n]

    plt.imshow(test1)
    plt.show()    

    grayim = rgb2gray(test1)
    grayim = transform.resize(grayim,(40,40))

    (t1_feat, hogImage) = feature.hog(grayim, orientations=9, pixels_per_cell=(4, 4),
        cells_per_block=(2, 2), transform_sqrt=True, visualise=True)

    t1_feat.shape
    t1_predict = rf.predict(t1_feat)
    print("==========")
    print("True :{}\npredicted:{}\n".format(t1_true,t1_predict[0]))
    # show the prediction
    print("I think that belongs to class: {}".format(t1_predict[0]))
    print("==========")
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)
    img = Image.fromarray(test1)
    draw = ImageDraw.Draw(img)
    draw.text((0,0), str(t1_predict), (0,255,0), font=font)
    draw = ImageDraw.Draw(img)
    plt.show(img) 
```


```python
for i in range(10):
    testing_on_data()
```


![png](output_37_0.png)


    ==========
    True :5.0
    predicted:12.0
    
    I think that belongs to class: 12.0
    ==========



![png](output_37_2.png)


    ==========
    True :5.0
    predicted:5.0
    
    I think that belongs to class: 5.0
    ==========



![png](output_37_4.png)


    ==========
    True :1.0
    predicted:1.0
    
    I think that belongs to class: 1.0
    ==========



![png](output_37_6.png)


    ==========
    True :2.0
    predicted:2.0
    
    I think that belongs to class: 2.0
    ==========



![png](output_37_8.png)


    ==========
    True :1.0
    predicted:1.0
    
    I think that belongs to class: 1.0
    ==========



![png](output_37_10.png)


    ==========
    True :0.0
    predicted:0.0
    
    I think that belongs to class: 0.0
    ==========



![png](output_37_12.png)


    ==========
    True :4.0
    predicted:4.0
    
    I think that belongs to class: 4.0
    ==========



![png](output_37_14.png)


    ==========
    True :1.0
    predicted:1.0
    
    I think that belongs to class: 1.0
    ==========



![png](output_37_16.png)


    ==========
    True :2.0
    predicted:2.0
    
    I think that belongs to class: 2.0
    ==========



![png](output_37_18.png)


    ==========
    True :1.0
    predicted:1.0
    
    I think that belongs to class: 1.0
    ==========



```python

```
