#Based on paper Comparing two deep learning sequence-based models for protein-protein interaction prediction by Richoux, Servantie, Bores, and Teletchea
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


class RichouxFullNetwork(nn.Module):
	def __init__(self,proteinLength=1166,encodingSize=24,linearSize=20,seed=1,):
		super(RichouxFullNetwork, self).__init__()
		
		torch.manual_seed(seed)

		
		self.linear1 = nn.Linear(proteinLength*encodingSize,linearSize)
		self.batchNorm1 = nn.BatchNorm1d(linearSize)
		self.linear2 = nn.Linear(linearSize,linearSize)
		self.batchNorm2 = nn.BatchNorm1d(linearSize)
		self.linear3 = nn.Linear(linearSize*2,linearSize)
		self.batchNorm3 = nn.BatchNorm1d(linearSize)
		self.linear4 = nn.Linear(linearSize,2)
		self.sigmoid = nn.Sigmoid()

		self.apply(self.weight_init)
		
	def weight_init(self,layer):
		if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv1d):
			torch.nn.init.xavier_uniform_(layer.weight)
		if isinstance(layer,nn.LSTM):
			for i in range(0,layer.num_layers):
				torch.nn.init.xavier_uniform_(layer._parameters['weight_ih_l'+str(i)])
				torch.nn.init.xavier_uniform_(layer._parameters['weight_hh_l'+str(i)])
		
	def forward(self,x):
		(protA, protB) = x
		protLst = []
		#run each protein through first few layers
		for item in [protA, protB]:
			#flatten proteins to single vector of data per protein in batch
			item = item.view(item.shape[0],-1)
			item = self.linear1(item)
			item = self.batchNorm1(item)
			item = self.linear2(item)
			item = self.batchNorm2(item)
			protLst.append(item)
			
		protA = protLst[0]
		protB = protLst[1]
		x = torch.cat((protA,protB),dim=1)
		x = self.linear3(x)
		x = self.batchNorm3(x)
		x = self.linear4(x)
		#x = self.sigmoid(x)
		#x = x.squeeze(1)
		return x

class RichouxLSTMNetwork(nn.Module):
	def __init__(self,encodingSize=24,convOutChannels=5,convKernelSize=20,hiddenSize=32,finalSize=25,seed=1):
		super(RichouxLSTMNetwork, self).__init__()
		
		torch.manual_seed(seed)

		self.pooling = nn.MaxPool1d(3)
		self.convLst = nn.ModuleList()
		self.batchNormLst = nn.ModuleList()
		for i in range(0,3):
			self.convLst.append(nn.Conv1d(encodingSize if i ==0 else convOutChannels, convOutChannels, convKernelSize))
			self.batchNormLst.append(nn.BatchNorm1d(convOutChannels))
		
		self.LSTM = nn.LSTM(input_size=convOutChannels,hidden_size=hiddenSize,batch_first=True)
		#self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		
		self.linear1 = nn.Linear(hiddenSize*2,finalSize)
		self.batchNorm1 = nn.BatchNorm1d(finalSize)
		self.linear2 = nn.Linear(finalSize,2)
		
		self.apply(self.weight_init)
		
	def weight_init(self,layer):
		if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv1d):
			torch.nn.init.xavier_uniform_(layer.weight)
		if isinstance(layer,nn.LSTM):
			for i in range(0,layer.num_layers):
				torch.nn.init.xavier_uniform_(layer._parameters['weight_ih_l'+str(i)])
				torch.nn.init.xavier_uniform_(layer._parameters['weight_hh_l'+str(i)])
				
	def forward(self,x):
		(protA, protB) = x
		protLst = []
		for item in [protA, protB]: #run each protein through LSTM/conv/pooling layers
			for i in range(0,3):
				#convolution on dim2, reducing/keeping channels (dim 1) at 5 (starts at 24)
				item = self.convLst[i](item)
				#apply relu
				item = self.relu(item)
				#max pool over sequence length (dim 2), dividing it by 3
				item = self.pooling(item)
				#batch norm over 5 channels
				item = self.batchNormLst[i](item)
		
			#flip axes to make hidden data (size 5, current in dim 2) last 
			item = item.permute(0,2,1)
			#grab the last set of hidden values for each item in the batch
			#pytorch specifically gives out this element (h_n) in addition to the standard output for each timestep. (This could also be accessed using output[0][:,-1,:] when batch is first)
			#call squeeze to remove first dimension (numer_layers is 1)
			item = self.LSTM(item)[1][0].squeeze(0)
			#pytorch applies tanh to output by default
		#	item = self.tanh(item)
			protLst.append(item)
			
		protA = protLst[0]
		protB = protLst[1]
		x = torch.cat((protA,protB),dim=1)
		x = self.linear1(x)
		x = self.relu(x)
		x = self.batchNorm1(x)
		x = self.linear2(x)
		return x
	
				

class RichouxModel(GenericNetworkModel):
	def __init__(self,hyp={},encodingSize=24,modelType='LSTM',maxProteinLength=1166,fullGPU=False,deviceType=None,numEpochs=100,batchSize=2048,lr=1e-3,minLr=8e-4,schedFactor=.9,schedPatience=3,schedThresh=1e-2):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,schedThresh=schedThresh)
		
		self.modelType = hyp.get('modelType',modelType).upper()
		if self.modelType not in ['LSTM','FULL']:
			self.modelType = 'LSTM'
		self.maxProteinLength = hyp.get('maxProteinLength',maxProteinLength)
		self.encodingSize = encodingSize #can not override through hyperparams, passed in from model, based on provided encoding files
		
	def genModel(self):
		if self.modelType == 'LSTM':
			self.net = RichouxLSTMNetwork(self.encodingSize,seed=self.seed)
		else:
			self.net = RichouxFullNetwork(self.maxProteinLength,self.encodingSize,seed=self.seed)
		self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
		
		
#protein length should be at least 3**3 to survive 3 sets of maxpool(3) layers
class RichouxNetworkModule(GenericNetworkModule):
	def __init__(self, hyperParams = {}, maxProteinLength=1166, modelType='LSTM',validationRate=0.1, encodingSize=24, seed=1):
		GenericNetworkModule.__init__(self,hyperParams)
		self.maxProteinLength = self.hyperParams.get('maxProteinLength',maxProteinLength)
		self.modelType = modelType
		self.validationRate = validationRate
		self.seed = self.hyperParams.get('seed',seed)
		self.encodingSize= self.hyperParams.get('encodingSize',encodingSize) #placeholder, will be overridden when loading data

	def genModel(self):
		self.model = RichouxModel(self.hyperParams,self.encodingSize,self.modelType,self.maxProteinLength)

	def loadFeatureData(self,featureFolder):
		self.dataLookup, self.dataMatrix = self.loadEncodingFileWithPadding(featureFolder+'OneHotEncoding24.encode',self.maxProteinLength)
		self.encodingSize = self.dataMatrix.shape[1]


#protein length should be at least 3**3 to survive 3 sets of maxpool(3) layers
class RichouxNetworkModuleLSTM(RichouxNetworkModule):
	def __init__(self, hyperParams = {}, maxProteinLength=1166, modelType='LSTM',validationRate=0.1, encodingSize=24, seed=1):
		super().__init__(hyperParams,maxProteinLength,modelType,validationRate,encodingSize,seed)
	
class RichouxNetworkModuleFULL(RichouxNetworkModule):
	def __init__(self, hyperParams = {}, maxProteinLength=1166, modelType='FULL',validationRate=0.1, encodingSize=24, seed=1):
		super().__init__(hyperParams,maxProteinLength,modelType,validationRate,encodingSize,seed)