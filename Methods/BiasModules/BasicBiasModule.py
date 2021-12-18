#Based on paper Using support vector machine combined with auto covariance to predict protein–protein interactions from protein sequences by Yanzhi Guo, Lezheng Yu, Zhining Wen, and Menglong Li

import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
from GenericModule import GenericModule
from GenericProteinScorer import GenericProteinScorer 
from collections import Counter

class BasicBiasModule(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		
		self.featureFolder = None
	def genModel(self,scoreData,mappingData):
		self.model = GenericProteinScorer(scoreData,mappingData)

	def fit(self,trainFeatures,trainClasses):
		scoreData = Counter()
		mappingData = {}
		for i in range(0,trainFeatures.shape[0]):
			curClass = trainClasses[i]*2-1 #convert to +1/-1
			for j in range(0,2):
				scoreData[str(trainFeatures[i,j])] += curClass
				mappingData[str(trainFeatures[i,j])] = [(str(trainFeatures[i,j]),1)]
		self.genModel(scoreData,mappingData) #create a new model from scratch, ensuring we don't overwrite the previously trained one
		
	def loadFeatureData(self,featureFolder):
		self.featureFolder = featureFolder
		
	#by default, load all data into a single 2D matrix, and return it with the class Data
	#if returnDict = True, returns dictionary instead
	def genFeatureData(self,pairs,dataType='train',returnDict=False):
		classData = pairs[:,2]
		classData = classData.astype(np.int)
		pairs = pairs[:,0:2]
		return (pairs,classData)

	#no scaling in this model
	def scaleFeatures(self,features,scaleType):
		return features
		
	def saveFeatScaler(self,fname):
		pass
			
	def loadFeatScaler(self,fname):
		pass
		
	def predictFromBatch(self,testPairs,batchSize,model=None):
		#no reason to use batches since pairwise data isn't created until dataloader
		return self.predictPairs(testPairs,model)
		
	#no reason to load pairs from file for this model
	def predictFromFile(self,testFile,batchSize,sep='\t',headerLines=1,classIdx=-1,model=None):
		pass
