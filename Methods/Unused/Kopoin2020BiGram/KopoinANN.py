#Based on paper A Feature Extraction Method in Large Scale Prediction of Human Protein-Protein Interactions using Physicochemical Properties into Bi-gram by Charlemenge N'Diffon Kopoin, NTakpe Tchimou, Bernard Kouassi Saha, and Michel Babri



import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from GenericMethod import GenericMethod

import PPIPUtils
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
import torch
from LiModels import *
from NetworkRunner import NetworkRunner
from SimpleDataset import SimpleDataset
import torch
import torch.nn as nn


class KopoinANNNetwork(nn.Module):
	def __init__(self, featShape):
		super(KopoinANNNetwork, self).__init__()
		self.featShape = featShape
		self.act = nn.Sigmoid()
		self.layer0 = nn.Linear(featShape,featShape//2)
		self.layer1 = nn.Linear(featShape//2,featShape//2)
		self.layer2 = nn.Linear(featShape//2,2)
		
	def forward(self,x):
		x = self.layer0(x)
		x = self.act(x)
		x = self.layer1(x)
		x = self.act(x)
		x = self.layer2(x)
		return x



#hyperparams:
#fullGPU - transfer full training data onto gpu
#deviceType - cpu or cuda
#featShape - size of first layer, defaults to 800
#Note, the paper provided no optimization related information, so I'm just using the adam optimizer that I have as the network runner's default
class KopoinANNModel(object):
	def __init__(self,featShape=800,hyperParams={},fullGPU=False,deviceType='cpu',numEpochs=1000,batchSize=256):
		self.deviceType = deviceType
		self.numEpochs = numEpochs
		self.batchSize=batchSize
		self.seed = 1
		self.fullGPU=  fullGPU
		if 'fullGPU' in hyperParams:
			self.fullGPU = hyperParams['fullGPU']
		if 'deviceType' in hyperParams:
			self.deviceType = hyperParams['deviceType']
		if 'numEpochs' in hyperParams:
			self.numEpochs = int(hyperParams['numEpochs'])
		if 'batchSize' in hyperParams:
			self.batchSize = int(hyperParams['batchSize'])
		if 'seed' in hyperParams:
			self.seed = int(hyperParams['seed'])
			
		if self.deviceType not in ['cpu','cuda']:
			self.deviceType = 'cpu'
		if self.fullGPU not in [True, False]:
			self.fullGPU = False
			
		self.model = None
		self.featShape = None

	def saveModel(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		self.model.save(fname)
		
	def genModel(self, featShape):
		self.model =NetworkRunner(KopoinANNNetwork(featShape),self.batchSize,self.deviceType)
		self.featShape = featShape
		
	def loadModel(self,fname,featureShape=800):
		if self.model is None or self.featShape != featureShape:
			self.genModel(featureShape)
		self.model.load(fname)

	#train network
	def fit(self,features,classes):
		self.genModel(features[0].shape[0])
		self.model.trainNumpy(features,classes,self.numEpochs,self.seed,full_gpu=self.fullGPU)
		
	#predict on network
	def predict_proba(self,features):
		probs,loss = self.model.predictNumpy(features)
		return probs
		

class KopoinANN(GenericMethod):
	def __init__(self, hyperParams = None):
		GenericMethod.__init__(self,hyperParams)
		
	def genModel(self):
		featShape = 800
		if self.featuresData is not None:
			featShape = self.featuresData.data.shape[0]*2 #2 for 2 proteins
		if 'featuresShape' in self.hyperParams:
			featShape = self.hyperParams['featuresShape'] #override using hyperparams if provided
		self.model = KopoinANNModel(featShape,self.hyperParams)

	def saveModelToFile(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		else:
			self.model.saveModel(fname)
		
	def loadModelFromFile(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		else:
			featShape = 800
			if self.featuresData is not None:
				featShape = self.featuresData.data.shape[0]*2 #2 for 2 proteins
			self.model.loadModel(fname,featShape)
		
	def loadFeatureData(self,featureFolder):
		print('loading')
		self.featuresData = ProteinFeaturesHolder([featureFolder+ 'KopoinBiGram.tsv'])
		print('done loading')
		
	def genFeatureData(self,pairs,dataType='Train'):	
		classData = pairs[:,2]
		featsData = self.featuresData.genData(pairs)
		return featsData, classData
	