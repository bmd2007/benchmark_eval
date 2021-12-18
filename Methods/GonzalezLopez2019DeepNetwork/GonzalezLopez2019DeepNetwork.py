#Based on paper Prediction of Protein-Protein Interactions Using An Effective Sequence Based Combined Method by Gonzalez-Lopez, Morales-Cordovilla, Villegas-Morcillo, Gomez, and Sanchez
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
from GenericNetworkModule import GenericNetworkModule
from GenericNetworkModel import GenericNetworkModel
import torch
import torch.nn as nn
from joblib import dump, load



class GonzalezLopez2019Network(nn.Module):
	def __init__(self,uniqueEmbeddings=8001,embedSize=512,hiddenSize=64,maxProteinLength=1000,seed=1):
		super(GonzalezLopez2019Network, self).__init__()

			
		torch.manual_seed(seed)
		self.embedding = nn.Embedding(uniqueEmbeddings,embedSize)
		
		self.protBatchNorm = nn.ModuleList()
		self.protGRU = nn.ModuleList()
		self.protLinear = nn.ModuleList()
		self.protBatchNorm2 = nn.ModuleList()
		
		self.dropout = nn.Dropout(0.5)
		
		for i in range(0,2):
			self.protBatchNorm.append(nn.BatchNorm1d(maxProteinLength))
			self.protGRU.append(nn.GRU(input_size=embedSize,hidden_size=hiddenSize,batch_first=True))
			self.protLinear.append(nn.Linear(hiddenSize,hiddenSize,bias=False))
			self.protBatchNorm2.append(nn.BatchNorm1d(hiddenSize))
		

		self.linear1 = nn.Linear(hiddenSize*2,embedSize,bias=False)
		self.batchNorm1 = nn.BatchNorm1d(embedSize)
		self.linear2 = nn.Linear(embedSize,2)

		self.act = nn.ELU()
		
	def forward(self,x):
		(protA, protB) = x
		protLst = [protA,protB]
		for i in range(0,2):
			item = protLst[i].long()
			item = self.embedding(item)
			item = self.protBatchNorm[i](item)
			item,hidden = self.protGRU[i](item)
			item = hidden.squeeze(0)
			item = self.protLinear[i](item)
			item = self.protBatchNorm2[i](item)
			item = self.act(item)
			item = self.dropout(item)
			
			protLst[i] = item
			
		protA = protLst[0]
		protB = protLst[1]
		x = torch.cat((protA,protB),dim=1)
		x = self.linear1(x)
		x = self.batchNorm1(x)
		x = self.act(x)
		x = self.linear2(x)
		return x
	
				

class GonzalezLopez2019Model(GenericNetworkModel):
	def __init__(self,hyp={},uniqueEmbeddings=8001,embedSize=512,maxProteinLength=1000,hiddenSize=64,fullGPU=False,deviceType=None,numEpochs=100,batchSize=256,lr=1e-2,minLr=2e-3,schedFactor=.5,schedPatience=2,schedThresh=1e-2,threshSchedMode='abs',optType='RMSprop'):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,schedThresh=schedThresh,optType=optType)
		
		self.uniqueEmbeddings = uniqueEmbeddings #can not override through hyperparams, passed in from model, based on provided encoding files
		self.embedSize= hyp.get('embedSize',embedSize)
		self.hiddenSize = hyp.get('hiddenSize',hiddenSize)
		self.maxProteinLength = maxProteinLength #can not override through hyperparams, passed in from model, based on provided encoding files
		hyp['threshSchedMode'] = hyp.get('threshSchedMode',threshSchedMode)
		
	def genModel(self):
		self.net = GonzalezLopez2019Network(self.uniqueEmbeddings,self.embedSize,self.hiddenSize,self.maxProteinLength,self.seed)
		self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
		
		
class GonzalezLopez2019Module(GenericNetworkModule):
	def __init__(self, hyperParams = {}, validationRate=0.1, maxProteinLength=1000, uniqueEmbeddings=8001, seed=1):
		GenericNetworkModule.__init__(self,hyperParams)
		self.maxProteinLength = self.hyperParams.get('ValidationRate',validationRate)
		self.maxProteinLength = self.hyperParams.get('maxProteinLength',maxProteinLength)
		self.seed = self.hyperParams.get('seed',seed)
		self.uniqueEmbeddings = self.hyperParams.get('uniqueEmbeddings',uniqueEmbeddings)#placeholder, will be overridden when loading data
		
	def genModel(self):
		self.model = GonzalezLopez2019Model(self.hyperParams,self.uniqueEmbeddings, self.maxProteinLength)

	def loadFeatureData(self,featureFolder):
		self.dataLookup, self.dataMatrix, lookupMatrix = self.loadEncodingFileWithPadding(featureFolder+'NumericEncoding20Skip3.encode',self.maxProteinLength, 'left',True)
		self.dataMatrix = self.dataMatrix.long().squeeze(1)
		self.uniqueEmbeddings = lookupMatrix.max().item()+1
