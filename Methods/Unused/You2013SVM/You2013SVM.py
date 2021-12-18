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
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from GenericSVM import GenericSVM



class You2013SVM(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		#self.scaler = StandardScaler()
		self.MRMR = MIMR() #no mrm algorithms to use :(
		self.modelType = None
		
	def genModel(self):
		self.model = GenericSVM(self.hyperParams)
				
	def saveModelToFile(self,fname):
		self.model.saveModelToFile(fname)

	def loadModelFromFile(self,fname):
		if self.model is None:
			self.genModel()
		self.model.loadModelFromFile(fname)
		
	def loadFeatureData(self,featureFolder):
		self.featuresData = {}
		self.featuresData['all'] = ProteinFeaturesHolder([featureFolder + 'MCD4_CTD_ConjointTriad_C.tsv',featureFolder + 'MCD4_CTD_ConjointTriad_T.tsv',featureFolder + 'MCD4_CTD_ConjointTriad_D.tsv'])
		
	def genFeatureData(self,pairs,dataType='train'):
		classData = pairs[:,2]
		classData = classData.astype(np.int)
		featData = self.featuresData['all'].genData(pairs)
		if dataType == 'train':
			print(featData.shape)
			featData = self.MRMR.fit_transform(featData,classData)
			print(featData.shape)
		else:
			featData = self.MRMR.transform(featData,classData)
		return featData, classData
	