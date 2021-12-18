#Based on paper Multifaceted protein–protein interaction prediction based on Siamese residual RCNN by Chen, Ju, Zhou, Chen, Zhang, Chang, Zaniolo, and Wang
#https://github.com/muhaochen/seq_ppi

import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import PPIPUtils
import time
import numpy as np
import torch
from NetworkRunnerCollate import NetworkRunnerCollate
from SimpleTorchDictionaryDataset import SimpleTorchDictionaryDataset
from GenericNetworkModel import GenericNetworkModel
from GenericNetworkModule import GenericNetworkModule
import torch
import torch.nn as nn
from joblib import dump, load

class ChenNetwork(nn.Module):
	def __init__(self,hiddenSize=50,inSize=14,numLayers=6,seed=1):
		super(ChenNetwork, self).__init__()
		
		
		self.pooling = nn.MaxPool1d(3)
		self.activation = nn.LeakyReLU(0.3)
		self.numLayers = numLayers
		torch.manual_seed(seed)

		self.convLst = nn.ModuleList()
		self.GRULst = nn.ModuleList()
		for i in range(0,self.numLayers):
			if i == 0: #first convolutions takes data of input size, other 5 take data of hidden size * 3
				self.convLst.append(nn.Conv1d(inSize,hiddenSize,3))
			else:
				self.convLst.append(nn.Conv1d(hiddenSize*3,hiddenSize,3))
			if i<= 4: #only numlayers-1 grus
				self.GRULst.append(nn.GRU(input_size=hiddenSize,hidden_size=hiddenSize,bidirectional=True,batch_first=True))
		
		self.linear1 = nn.Linear(hiddenSize,100)
		self.linear2 = nn.Linear(100,(hiddenSize+7)//2)
		self.linear3 = nn.Linear((hiddenSize+7)//2,2)
		
	def forward(self,x):
		(protA, protB) = x
		protLst = []
		for item in [protA, protB]: #run each protein through gru/pooling layers
			for i in range(0,self.numLayers-1):
				#conv1d and pooling expect hidden dim on 2nd axis (dim=1), gru needs hidden dim on 3rd axis (dim=2) . . .
				item = item.permute(0,2,1)
				item = self.convLst[i](item)
				item = self.pooling(item)
				item = item.permute(0,2,1)
				item2,hidden = self.GRULst[i](item)
				item = torch.cat((item,item2),2)
			
			item = item.permute(0,2,1)
			item = self.convLst[self.numLayers-1](item)
			item = item.mean(dim=2) #global average pooling over dim 2, reducing the data from 3D to 2D
			protLst.append(item)
		
		protA = protLst[0]
		protB = protLst[1]
		x = torch.mul(protA,protB) #element wise multiplication
		
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.linear3(x)
		return x
	
class NetworkRunnerChen(NetworkRunnerCollate):
	def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=1,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=True,hyp={},skipScheduler=30):
		NetworkRunnerCollate.__init__(self,net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax,hyp)
		self.skipScheduler = hyp.get('skipScheduler',skipScheduler)
	
			
	def updateScheduler(self,values):
		if self.scheduler is not None and self.epoch > self.skipScheduler:
			self.scheduler.step(values)

class ChenModel(GenericNetworkModel):
	def __init__(self,hyp={},inSize=12,hiddenSize=50,numLayers=6,fullGPU=False,deviceType=None,numEpochs=100,batchSize=256,lr=5e-4,minLr=1e-4,schedFactor=.5,schedPatience=3,schedThresh=1e-2,threshSchedMode='abs'):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,schedThresh=schedThresh)
		
		self.inSize = inSize
		self.hiddenSize = hyp.get('hiddenSize',hiddenSize)
		self.numLayers = hyp.get('numLayers',numLayers)
		
		#move uncommon network runner properties into hyperparams list if needed
		hyp['amsgrad'] = hyp.get('amsgrad',True)
		hyp['threshSchedMode'] = hyp.get('threshSchedMode',threshSchedMode)
		
		
	def genModel(self):
		self.net = ChenNetwork(self.hiddenSize,self.inSize,self.numLayers,self.seed)
		#self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
		self.model = NetworkRunnerChen(self.net,hyp=self.hyp,skipScheduler=self.skipScheduler)

	#train network
	def fit(self,pairLst,classes,dataMatrix,validationPairs=None, validationClasses=None):
		self.skipScheduler = 250000//classes.shape[0]
		super().fit(pairLst,classes,dataMatrix,validationPairs,validationClasses)
		
				
#protein length should be at least 3**5 to survive 5 sets of maxpool(3) layers
class ChenNetworkModule(GenericNetworkModule):
	def __init__(self, hyperParams = {}, maxProteinLength=2000, hiddenSize=50,inSize=12):
		GenericNetworkModule.__init__(self,hyperParams)
		self.maxProteinLength = self.hyperParams.get('maxProteinLength',maxProteinLength)
		self.inSize = self.hyperParams.get('inSize',inSize) #temporary value, until data is loaded in loadFeatureData function
		self.hiddenSize = self.hyperParams.get('hiddenSize',hiddenSize)
		
	def genModel(self):
		self.model = ChenModel(self.hyperParams,self.inSize,self.hiddenSize)
		
	def loadFeatureData(self,featureFolder):
		dataLookupSkip, dataMatrixSkip = self.loadEncodingFileWithPadding(featureFolder+'SkipGramAA7H5.encode',self.maxProteinLength)
		dataLookupOneHot, dataMatrixOneHot = self.loadEncodingFileWithPadding(featureFolder+'OneHotEncoding7.encode',self.maxProteinLength)

		allProteinsSet = set(list(dataLookupSkip.keys())) & set(list(dataLookupOneHot.keys()))

		self.encodingSize = dataMatrixSkip.shape[1] + dataMatrixOneHot.shape[1]

		self.dataLookup = {}
		self.dataMatrix = torch.zeros((len(allProteinsSet),self.maxProteinLength,self.encodingSize))
		for item in allProteinsSet:
			self.dataLookup[item] = len(self.dataLookup)
			skipData = dataMatrixSkip[dataLookupSkip[item],:,:].T
			oneHotData = dataMatrixOneHot[dataLookupOneHot[item],:,:].T
			self.dataMatrix[self.dataLookup[item],:,:skipData.shape[1]] = skipData
			self.dataMatrix[self.dataLookup[item],:,skipData.shape[1]:] = oneHotData
			
		
		
		

	
		
		