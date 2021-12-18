#Code based on python DScript library (pip install dscript )
#This file mostly uses the hyperparameters and sets up the networks as done in the train.py file from dscript
#MIT License

#Copyright (c) 2020 Samuel Sledzieski

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


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
from NetworkRunnerSledzieski import NetworkRunnerSledzieski
from SledzieskiDataset import SledzieskiDataset
import torch
import torch.nn as nn

from dscript.models.embedding import FullyConnectedEmbed
from dscript.models.contact import ContactCNN
from dscript.models.interaction import ModelInteraction


class SledzieskNetwork(nn.Module):
	def __init__(self,projDim,dropout,hidDim,kernelWidth,useW,poolWidth,seed=1,hyp={}):
		super().__init__()
		
		projDim = hyp.get('projDim',projDim)
		dropOut = hyp.get('dropOut',dropOut)
		hidDim = hyp.get('hidDim',hidDim)
		kernelWidth = hyp.get('kernelWidth',kernelWidth)
		poolWidth = hyp.get('poolWidth',poolWidth)
		useWeight = hyp.get('useWeight',useWeight)
		seed = hyp.get('seed',seed)
		
		torch.manual_seed(seed)
		
		self.embedding = FullyConnectedEmbed(6165, projDim, dropout=dropout)
		self.contactCNN = ContactCNN(projDim,hidDim,kernelWidth)
		self.modelInteraction = ModelInteraction(self.embedding,self.contactCNN,pool_size=poolWidth,use_W=useW)
		
		
	def forward(self,x,deviceType='cpu'):
		cmLst = []
		phLst = []
		for i in range(0,len(x[0])):
			a = x[0][i].to(deviceType).unsqueeze(0)
			b = x[1][i].to(deviceType).unsqueeze(0)
			cm, ph = self.modelInteraction.map_predict(a,b)
			cmLst.append(torch.mean(cm))
			phLst.append(ph)
		return torch.vstack(cmLst), torch.vstack(phLst)


    

#hyperparams:
#folderName -- folder containing protein data for network
#deviceType - cpu or cuda
#dropOut -- drop out in embedding layer
#projDim -- Dimension of projection layer
#hidDim -- Dimension of hidden layer
#kernelWidth -- Width of convolutional kernel-width
#poolWidth -- width of max pool in interaction module
#useWeight -- Use weight matrix in interaction prediction model
#simObjective -- Weight of similarity objective
#pairAugment -- create pair (A,B) for every pair (B,A)
class SledzieskiModel(object):
	def __init__(self,hyp={},folderName=None,deviceType=None,dropOut=0.5,projDim=100,hidDim=50,kernelWidth=7,poolWidth=9,useWeight=True,simObjective=0.35,pairAugment=False,numEpochs=100,batchSize=25,weightDecay=0.001,lr=0.001,sched_factor=0.4,sched_thresh=5e-3,minLr=1e-5):
	
		self.hyp = hyp
		self.numEpochs = hyp.get('numEpochs',numEpochs)
		self.seed = hyp.get('seed',1)
		hyp['seed'] = self.seed
		self.minLr = hyp.get('minLr',minLr)
		self.folderName = hyp.get('folderName',folderName)
		self.pairAugment = hyp.get('pairAugment',pairAugment)
				
		#move network runner properties into hyperparams list if needed
		hyp['weightDecay'] = hyp.get('weightDecay',weightDecay)
		hyp['batchSize'] = hyp.get('batchSize',batchSize)
		hyp['schedThresh'] = hyp.get('schedThresh',sched_thresh)
		hyp['schedFactor'] = hyp.get('schedFactor',sched_factor)
		hyp['simObjective'] = hyp.get('simObjective',simObjective)
		hyp['deviceType'] = hyp.get('deviceType',deviceType)
		hyp['lr'] = hyp.get('lr',lr)
		hyp['optType'] = hyp.get('optType','Adam')
				
		#move network properties into hyperparams as needed
		hyp['projDim'] = hyp.get('projDim',projDim)
		hyp['dropOut'] = hyp.get('dropOut',dropOut)
		hyp['hidDim'] = hyp.get('hidDim',hidDim)
		hyp['kernelWidth'] = hyp.get('kernelWidth',kernelWidth)
		hyp['poolWidth'] = hyp.get('poolWidth',poolWidth)
		hyp['useWeight'] = hyp.get('useWeight',useWeight)
		
		self.model = None
		
	def saveModel(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		self.model.save(fname)
		
	def genModel(self):
		self.net = SledzieskNetwork(hyp=self.hyp)
		self.model = NetworkRunnerSledzieski(self.net,hyp=self.hyp)
		
	def loadModel(self,fname):
		if self.model is None:
			self.genModel()
		self.model.load(fname)

	#train network
	def fit(self,features,classes):
		pairLst = features
		#negative/positive
		self.negativeRatio = round((classes.shape[0]-np.sum(classes))/np.sum(classes))
		if self.pairAugment:
			#copy/flip.
			features2 = np.hstack((np.expand_dims(features[:,1],0).T,np.expand_dims(features[:,0],0).T))
			features = np.vstack((features,features2))
			
		self.genModel()
		
		dataset = SledzieskiDataset(self.folderName,pairLst,classes)
		self.model.train(dataset,self.numEpochs,self.seed,min_lr=self.minLr)
		
	#predict on network
	def predict_proba(self,features):
		pairLst = features
		dataset = SledzieskiDataset(self.folderName,pairLst)
		
		probs,loss = self.model.predict(features,self.folderName)
		return probs
		



    

#hyperparams:
#folderName -- folder containing protein data for network
#deviceType - cpu or cuda
#dropOut -- drop out in embedding layer
#projDim -- Dimension of projection layer
#hidDim -- Dimension of hidden layer
#kernelWidth -- Width of convolutional kernel-width
#poolWidth -- width of max pool in interaction module
#useWeight -- Use weight matrix in interaction prediction model
#simObjective -- Weight of similarity objective
#pairAugment -- create pair (A,B) for every pair (B,A)
		

class SledzieskiDeepNetwork(GenericMethod):
	def __init__(self, hyperParams = None):
		GenericMethod.__init__(self,hyperParams)
		self.folderName = None
		
	def genModel(self):
		self.model = SledzieskiModel(self.hyperParams,self.folderName)

	def saveModelToFile(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		else:
			self.model.saveModel(fname)
		
	def loadModelFromFile(self,fname):
		if self.model is None:
			self.genModel()
		self.model.loadModel(fname)
		
	def loadFeatureData(self,featureFolder):
		#no feature loading, data stays in files
		self.folderName= featureFolder+'Berger/'
		
	def genFeatureData(self,pairs,dataType='Train'):	
		classData = pairs[:,2]
		featsData = pairs[:,0:2] #no feature data load, data stays in files
		return featsData, classData

	#no feature scaling, data is in files
	def scaleFeatures(self,features,scaleType):
		return features

	def setScaleFeatures(self,trainPairs):
		return


	def predictFromBatch(self,testPairs,batchSize,model=None):
		#no reason to use batches since pairwise data isn't created until dataloader
		return self.predictPairs(testPairs,model)
		
	#no reason to load pairs from file for this model
	def predictFromFile(self,testFile,batchSize,sep='\t',headerLines=1,classIdx=-1,model=None):
		pass