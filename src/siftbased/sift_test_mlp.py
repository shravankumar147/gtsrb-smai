

## TESTING
import cv2
import utils 
import pdb
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier



clf, classNames, stdScaler, k, voc = joblib.load("bagOfFeatures_MLP.pkl")


# create feature extractor and keypoint detector objects

featuresDetector = cv2.FeatureDetector_create('SIFT')
descriptorExtractor = cv2.DescriptorExtractor_create('SIFT')


testingPath = "../../GTSRB/train/Final_Training/Images/"
testingNames = os.listdir(testingPath)

imagesPaths= []

imageClasses = []

classId = 0

for testingName in testingNames:
	directory = os.path.join(testingPath, testingName)
	classPath = utils.imlist(directory)
	imagesPaths += classPath
	imageClasses += [classId]*len(classPath)
	classId += 1


# pdb.set_trace()
imageClassesList = []
descriptorAllList = []

for imagePath, imageClass in zip(imagesPaths, imageClasses):
	#print imagePath
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (40,40))
	features = featuresDetector.detect(image)
	features, descriptors = descriptorExtractor.compute(image, features)
	if descriptors is not None:
         descriptorAllList.append((imagePath, descriptors))
         imageClassesList.append(imageClass)


# Stack all the descriptors vertically
imageClassesAll = np.vstack(imageClassesList)
descriptorAll = np.vstack(zip(*descriptorAllList)[1])
print "Descriptor Size: %s" %(descriptorAll.shape, )
print "Image classes Size: %s" %(imageClassesAll.shape, )


# Calculate Image Features

testImageFeatures = np.zeros((len(descriptorAllList), k), "float32")
for i in range(len(descriptorAllList)):
	words, distance = vq(descriptorAllList[i][1], voc)
	for w in words:
		testImageFeatures[i, w] += 1 
print "Image Feature Size: %s" %(testImageFeatures.shape,)


# Scale the features
testImageFeatures = stdScaler.transform(testImageFeatures)

# Perform the predictions
predictions = [i for i in clf.predict(testImageFeatures)]

predictions = np.asarray(predictions)
imageClasses = np.asarray(imageClassesList)

print "Accuracy: %f" %accuracy_score(imageClasses, predictions)

pdb.set_trace()
