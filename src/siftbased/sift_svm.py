
# import argparse as ap
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


# create feature extractor and keypoint detector objects

featuresDetector = cv2.FeatureDetector_create('SIFT')
descriptorExtractor = cv2.DescriptorExtractor_create('SIFT')

# This the list where all descriptor will be stored
descriptorAllList = []

trainingPath = "../../GTSRB/train/Final_Training/Images/"
trainingNames = os.listdir(trainingPath)

imagesPaths= []

imageClasses = []

classId = 0

for trainingName in trainingNames:
	directory = os.path.join(trainingPath, trainingName)
	classPath = utils.imlist(directory)
	imagesPaths += classPath
	imageClasses += [classId]*len(classPath)
	classId += 1

# pdb.set_trace()
imageClassesList = []
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

#pdb.set_trace()


# Perform k-means clustering
k = 400
voc, variance = kmeans(descriptorAll, k, 1) 

# Calculate Image Features

imageFeatures = np.zeros((len(descriptorAllList), k), "float32")
for i in range(len(descriptorAllList)):
	words, distance = vq(descriptorAllList[i][1], voc)
	for w in words:
		imageFeatures[i, w] += 1 
print "Image Feature Size: %s" %(imageFeatures.shape,)


stdScaler = StandardScaler().fit(imageFeatures)
imageFeatures = stdScaler.transform(imageFeatures)


# Train a Linear SVM

clf = LinearSVC()
clf.fit(imageFeatures, np.array(imageClassesAll))

# Save SVM

joblib.dump((clf, trainingNames, stdScaler, k, voc), "bagOfFeatures_SVM.pkl", compress = 3)








