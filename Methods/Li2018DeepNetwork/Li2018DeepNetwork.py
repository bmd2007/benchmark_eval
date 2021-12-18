#Based on paper Deep Neural Network Based Predictions of Protein Interactions Using Primary Sequences by Li, Gong, Yu, and Zhou
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



class Li2018Network(nn.Module):
	def __init__(self,uniqueEmbeddings=23,embedSize=128,numFilters=64,hiddenSize=80,seed=1):
		super(Li2018Network, self).__init__()
		
		torch.manual_seed(seed)
		self.embedding = nn.Embedding(22,embedSize)
		#note, in paper, convolution section of methods states 10 filters, and shows an image with 10
		#which is also stated in the hyperparameters section.
		#However, in the hyperparameters section, the outputs size is listed as 64 for each layers
		#I'm using 64 filters, as that would create the output size of 64 from each layers, by default
		
		self.pooling = nn.MaxPool1d(2)
		self.convLst = nn.ModuleList()
		self.relu = nn.ReLU()
		#self.tanh = nn.Tanh()
		
		for i in range(0,3):
			self.convLst.append(nn.Conv1d((embedSize if i == 0 else numFilters),numFilters,10))
		
		self.LSTM = nn.LSTM(input_size=numFilters,hidden_size=hiddenSize,batch_first=True)
		self.linear1 = nn.Linear(hiddenSize*2,2)
		
	def forward(self,x):
		(protA, protB) = x
		protLst = []
		for item in [protA, protB]: #run each protein through LSTM/conv/pooling layers
			item = item.long().squeeze(1) #concat all proteins in batch into single matrix
			item = self.embedding(item)
			item = item.permute(0,2,1)
			for i in range(0,3):
				#convolution on dim2, reducing/keeping channels (dim 1) at 5 (starts at 24)
				item = self.convLst[i](item)
				#apply relu
				item = self.relu(item)
				#max pool over sequence length (dim 2), dividing it by 2
				item = self.pooling(item)
		
			#flip axes to make hidden data (size 64, current in dim 2) last 
			item = item.permute(0,2,1)
			#grab the last set of hidden values for each item in the batch
			#pytorch specifically gives out this element (h_n) in addition to the standard output for each timestep. (This could also be accessed using output[0][:,-1,:] when batch is first)
			#call squeeze to remove first dimension (numer_layers is 1)
			item = self.LSTM(item)[1][0].squeeze(0)
			#pytorch applies tanh to output by default
			#item = self.tanh(item)
			protLst.append(item)
			
		protA = protLst[0]
		protB = protLst[1]
		x = torch.cat((protA,protB),dim=1)
		x = self.linear1(x)
		return x
	
				

class Li2018Model(GenericNetworkModel):
	def __init__(self,hyp={},uniqueEmbeddings=23,embedSize=128,numFilters=64,hiddenSize=80,fullGPU=False,deviceType=None,numEpochs=100,batchSize=256,lr=1e-3,minLr=2e-4,schedFactor=.5,schedPatience=1,schedThresh=2e-2,threshSchedMode='abs'):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,schedThresh=schedThresh)
		
		self.uniqueEmbeddings = uniqueEmbeddings #can not override through hyperparams, passed in from model, based on provided encoding files
		self.numFilters = hyp.get('numFilters',64)
		self.hiddenSize = hyp.get('hiddenSize',80)
		self.embedSize= hyp.get('embedSize',embedSize)
		hyp['threshSchedMode'] = hyp.get('threshSchedMode',threshSchedMode)
		
		
	def genModel(self):
		self.net = Li2018Network(self.uniqueEmbeddings,self.embedSize,self.numFilters,self.hiddenSize,self.seed)
		self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
		
class Li2018DeepNetworkModule(GenericNetworkModule):
	def __init__(self, hyperParams = {}, maxProteinLength=1200, uniqueEmbeddings=23, seed=1):
		GenericNetworkModule.__init__(self,hyperParams)
		self.maxProteinLength = self.hyperParams.get('maxProteinLength',maxProteinLength)
		self.seed = self.hyperParams.get('seed',seed)
		self.uniqueEmbeddings = self.hyperParams.get('uniqueEmbeddings',uniqueEmbeddings)#placeholder, will be overridden when loading data
		
	def genModel(self):
		self.model = Li2018Model(self.hyperParams,self.uniqueEmbeddings)

	def loadFeatureData(self,featureFolder):
		self.dataLookup, self.dataMatrix, lookupMatrix = self.loadEncodingFileWithPadding(featureFolder+'NumericEncoding22.encode',self.maxProteinLength, 'left',True)
		self.dataMatrix = self.dataMatrix.long()
		self.uniqueEmbeddings = lookupMatrix.max().item()+1