#Based on paper An integration of deep learning with feature embedding for protein-protein interaction prediction by Yao, Du, Diao, and Zhu
import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from GenericNetworkModule import GenericNetworkModule
from GenericNetworkModel import GenericNetworkModel
import PPIPUtils
import time
import numpy as np
import torch
from NetworkRunnerCollate import NetworkRunnerCollate
from SimpleTorchDictionaryDataset import SimpleTorchDictionaryDataset
import torch
import torch.nn as nn
from joblib import dump, load


class YaoNetwork(nn.Module):
	def __init__(self,encodingSize=20,proteinLength=850,maxLen=2048,seed=1):
		super(YaoNetwork, self).__init__()
		
		torch.manual_seed(seed)
		self.proteinLength = proteinLength
		self.encodingSize = encodingSize
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.5)


		#first protein
		self.linearLst1 = nn.ModuleList()
		self.batchNormLst1 = nn.ModuleList()
		
		self.linearLst1.append(nn.Linear(proteinLength*encodingSize,maxLen))
		self.batchNormLst1.append(nn.BatchNorm1d(maxLen))
		self.linearLst1.append(nn.Linear(maxLen,maxLen//2))
		self.batchNormLst1.append(nn.BatchNorm1d(maxLen//2))
		self.linearLst1.append(nn.Linear(maxLen//2,maxLen//4))
		self.batchNormLst1.append(nn.BatchNorm1d(maxLen//4))
		self.linearLst1.append(nn.Linear(maxLen//4,maxLen//16))
		self.batchNormLst1.append(nn.BatchNorm1d(maxLen//16))
		
		
		#second protein
		self.linearLst2 = nn.ModuleList()
		self.batchNormLst2 = nn.ModuleList()
			
		self.linearLst2.append(nn.Linear(proteinLength*encodingSize,maxLen))
		self.batchNormLst2.append(nn.BatchNorm1d(maxLen))
		self.linearLst2.append(nn.Linear(maxLen,maxLen//2))
		self.batchNormLst2.append(nn.BatchNorm1d(maxLen//2))
		self.linearLst2.append(nn.Linear(maxLen//2,maxLen//4))
		self.batchNormLst2.append(nn.BatchNorm1d(maxLen//4))
		self.linearLst2.append(nn.Linear(maxLen//4,maxLen//16))
		self.batchNormLst2.append(nn.BatchNorm1d(maxLen//16))
		
		
		
		#both proteins
				
		self.linear1 = nn.Linear(maxLen//8,8)
		self.batchNorm1 = nn.BatchNorm1d(8)
		self.linear2 = nn.Linear(8,2)
		
	def forward(self,x):
		(protA, protB) = x
		protA = protA.view(-1,self.proteinLength*self.encodingSize)
		protB = protB.view(-1,self.proteinLength*self.encodingSize)
		
		for i in range(0,len(self.linearLst1)):
			protA = self.linearLst1[i](protA)
			protA = self.activation(protA)
			protA = self.batchNormLst1[i](protA)
			protA = self.dropout(protA)
			
			protB = self.linearLst2[i](protB)
			protB = self.activation(protB)
			protB = self.batchNormLst2[i](protB)
			protB = self.dropout(protB)
			
		
		x = torch.cat((protA,protB),dim=1)
		x = self.linear1(x)
		x = self.activation(x)
		x = self.batchNorm1(x)
		x = self.dropout(x)
		x = self.linear2(x)
		return x

	
				

class Yao2019Model(GenericNetworkModel):
	def __init__(self,hyp={},encodingSize=20,maxProteinLength=850,maxLayerSize=2048,fullGPU=False,deviceType=None,numEpochs=100,batchSize=256,lr=1e-2,minLr=2e-4,schedFactor=.4,schedPatience=3,schedThresh=1e-2,weightDecay=1e-2,optType='SGD'):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,weightDecay=weightDecay,optType=optType)
		
		self.maxProteinLength = hyp.get('maxProteinLength',maxProteinLength)
		self.encodingSize = encodingSize #can not override through hyperparams, passed in from model, based on provided encoding files
		self.maxLayerSize = hyp.get('maxLayerSize',maxLayerSize)
		
	def genModel(self):
		self.net = YaoNetwork(self.encodingSize,self.maxProteinLength,self.maxLayerSize)
		self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)

		
		
#protein length should be at least 3**3 to survive 3 sets of maxpool(3) layers
class Yao2019NetworkModule(GenericNetworkModule):
	def __init__(self, hyperParams = {}, maxProteinLength=850, encodingSize=20, seed=1):
		GenericNetworkModule.__init__(self,hyperParams)
		self.maxProteinLength = self.hyperParams.get('maxProteinLength',maxProteinLength)
		self.seed = self.hyperParams.get('seed',seed)
		self.hyperParams['seed'] = self.seed
		self.encodingSize= self.hyperParams.get('encodingSize',encodingSize) #placeholder, will be overridden when loading data

	def genModel(self):
		self.model = Yao2019Model(self.hyperParams,self.encodingSize,self.maxProteinLength)

	def loadFeatureData(self,featureFolder):
		self.dataLookup, self.dataMatrix = self.loadEncodingFileWithPadding(featureFolder+'SkipGramAA25H20.encode',self.maxProteinLength)
		self.encodingSize = self.dataMatrix.shape[1]