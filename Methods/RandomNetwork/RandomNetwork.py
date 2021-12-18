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
from ProteinFeaturesHolder import ProteinFeaturesHolder

class RandomNetwork(nn.Module):
	def __init__(self,featureSize=500,dropout=0.1,firstLayer=512,seed=1):
		super(RandomNetwork, self).__init__()
		torch.manual_seed(seed)
		
		self.activation = nn.LeakyReLU(0.1)
		self.dropout = nn.Dropout(dropout)
		
		self.linearLayers = nn.ModuleList()
		self.linearLayers.append(nn.Linear(featureSize,firstLayer))
		curSize = firstLayer
		while curSize >= 12:
			newSize = curSize//4
			self.linearLayers.append(nn.Linear(curSize,newSize))
			curSize=newSize
		self.outputLinear = nn.Linear(curSize*2,2)
		
		
	def forward(self,x):
		(protA, protB) = x
		protLst = []
		for item in [protA, protB]: #run each protein through gru/pooling layers
			for i in range(0,len(self.linearLayers)):
				#conv1d and pooling expect hidden dim on 2nd axis (dim=1), gru needs hidden dim on 3rd axis (dim=2) . . .
				item = self.linearLayers[i](item)
				item = self.dropout(item)
				item = self.activation(item)
			protLst.append(item)
		
		protA = protLst[0]
		protB = protLst[1]
		x = torch.cat((protA,protB),dim=1)
		x = self.outputLinear(x)
		return x

class RandomModel(GenericNetworkModel):
	def __init__(self,hyp={},featureSize=500,dropout=0.1,firstLayer=512,fullGPU=False,deviceType=None,numEpochs=100,batchSize=256,lr=5e-4,minLr=1e-4,schedFactor=.5,schedPatience=3,schedThresh=1.5e-3,threshSchedMode='abs'):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,schedThresh=schedThresh)
		
		self.dropout = hyp.get('dropout',dropout)
		self.featureSize = featureSize
		self.firstLayer = hyp.get('firstLayer',firstLayer)
		hyp['threshSchedMode'] = hyp.get('threshSchedMode',threshSchedMode)
		
		
	def genModel(self):
		self.net = RandomNetwork(self.featureSize,self.dropout,self.firstLayer,self.seed)
		#self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
		self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)

#protein length should be at least 3**5 to survive 5 sets of maxpool(3) layers
class RandomNetworkModule(GenericNetworkModule):
	def __init__(self, hyperParams = {}, featureSize=500):
		GenericNetworkModule.__init__(self,hyperParams)
		self.featureSize = self.hyperParams.get('featureSize',featureSize) #temporary value, until data is loaded in loadFeatureData function
		self.featLst = self.hyperParams.get('featLst',{'all':['Random500.tsv']})
		
	def genModel(self):
		self.model = RandomModel(self.hyperParams,self.featureSize)
		
	def loadFeatureData(self,featureFolder):
		lst = []
		for item in self.featLst['all']:
			lst.append(featureFolder+item)
		featuresData = ProteinFeaturesHolder(lst,convertToInt=False)
		self.dataLookup = {}
		for item in featuresData.rowLookup:
			self.dataLookup[str(item)] = featuresData.rowLookup[item]
		self.dataMatrix = torch.tensor(featuresData.data)
		#self.dataMatrix = torch.tensor(featuresData.data)
		self.featureSize = self.dataMatrix.shape[1]
		