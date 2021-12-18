#Based on paper DeepPPI: Boosting Prediction of Protein-Protein Interactions with Deep Neural Networks by Du, Sun, Hu, Yao, Yan, and Zhang
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
from ProteinFeaturesHolder import ProteinFeaturesHolder
import torch
from NetworkRunnerCollate import NetworkRunnerCollate
from SimpleTorchDictionaryDataset import SimpleTorchDictionaryDataset
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler





#Note:  Paper doesn't mention where dropout is applied, assuming between all layers, after activation
class Du2017NetworkSep(nn.Module):
	def __init__(self,hyp={},featureSize=1164,layerSizes=[512,256,128,128],dropRate=0.2,seed=1):
		super(Du2017NetworkSep, self).__init__()
		
		if 'layerSizes' in hyp:
			layerSizes = hyp['layerSizes']
		if 'dropRate' in hyp:
			dropRate = hyp['dropRate']
		if 'seed' in hyp:
			seed = hyp['seed']
		
		torch.manual_seed(seed)
		
		self.featureSize = featureSize
		self.dropout = nn.Dropout(dropRate)
		self.act = nn.ReLU()
		
		#protein1
		self.LinearP1Lst = nn.ModuleList()
		#protein2
		self.LinearP2Lst = nn.ModuleList()
		layerSizes = [featureSize]+layerSizes
		for i in range(0,len(layerSizes)-2):
			self.LinearP1Lst.append(nn.Linear(layerSizes[i],layerSizes[i+1]))
			self.LinearP2Lst.append(nn.Linear(layerSizes[i],layerSizes[i+1]))
			
		#layers after merge
		self.linear1 = nn.Linear(layerSizes[-2]*2,layerSizes[-1])
		self.linear2 = nn.Linear(layerSizes[-1],2)
		
		self.apply(self.weight_init)
		
	def weight_init(self,layer):
		if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv1d):
			torch.nn.init.xavier_normal_(layer.weight)
		if isinstance(layer,nn.LSTM):
			for i in range(0,layer.num_layers):
				torch.nn.init.xavier_normal_(layer._parameters['weight_ih_l'+str(i)])
				torch.nn.init.xavier_normal_(layer._parameters['weight_hh_l'+str(i)])
				
	def forward(self,x):
		(p1,p2) = x
		for i in range(0,len(self.LinearP1Lst)):
			p1 = self.LinearP1Lst[i](p1)
			p1 = self.act(p1)
			p1 = self.dropout(p1)
			
			p2 = self.LinearP2Lst[i](p2)
			p2 = self.act(p2)
			p2 = self.dropout(p2)
			
		
		x = torch.cat((p1,p2),dim=1)
		x = self.linear1(x)
		x = self.act(x)
		x = self.dropout(x)
		x = self.linear2(x)
		return x
		
class Du2017NetworkComb(nn.Module):
	def __init__(self,hyp={},featureSize=2328,layerSizes=[512,256,128,128],dropRate=0.2,seed=1):
		super(Du2017NetworkComb, self).__init__()
		
		if 'layerSizes' in hyp:
			layerSizes = hyp['layerSizes']
		if 'dropRate' in hyp:
			dropRate = hyp['dropRate']
		if 'seed' in hyp:
			seed = hyp['seed']
		
		torch.manual_seed(seed)
		
		self.featureSize = featureSize
		self.dropout = nn.Dropout(dropRate)
		self.act = nn.ReLU()
		
		self.LayerLst = nn.ModuleList()
		layerSizes = [featureSize]+layerSizes+[2]
		for i in range(0,len(layerSizes)-1):
			self.LayerLst.append(nn.Linear(layerSizes[i],layerSizes[i+1]))
		

		self.apply(self.weight_init)
		
	def weight_init(self,layer):
		if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv1d):
			torch.nn.init.xavier_normal_(layer.weight)
		if isinstance(layer,nn.LSTM):
			for i in range(0,layer.num_layers):
				torch.nn.init.xavier_normal_(layer._parameters['weight_ih_l'+str(i)])
				torch.nn.init.xavier_normal_(layer._parameters['weight_hh_l'+str(i)])
				
	def forward(self,x):
		(p1,p2) = x
		x = torch.cat((p1,p2),dim=1)

		for i in range(0,len(self.LayerLst)):
			x = self.LayerLst[i](x)
			if i != len(self.LayerLst)-1:
				x = self.act(x)
				x = self.dropout(x)
		
		return x
		
		
class Du2017Model(GenericNetworkModel):
	def __init__(self,hyp={},modelType='SEP',featureSize=1164,fullGPU=False,deviceType=None,numEpochs=500,batchSize=64,lr=1e-2,schedFactor=0.4,schedThresh=3e-2,schedPatience=2,minLr=2e-4,optType='SGD',momentum=0.9):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,schedFactor=schedFactor,schedThresh=schedThresh,schedPatience=schedPatience,minLr=minLr,optType=optType,momentum=momentum)
		
		self.modelType = hyp.get('modelType',modelType).upper()
		if self.modelType not in ['SEP','COMB']:
			self.modelType = 'SEP'
		
		self.featureSize = featureSize #cannot be overriden, passed in by module after loading data
		
	def genModel(self):
		if self.modelType == 'SEP':
			self.net = Du2017NetworkSep(self.hyp,self.featureSize)
		if self.modelType == 'COMB':
			self.net = Du2017NetworkComb(self.hyp,self.featureSize)
		self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
		
	

class Du2017DeepNetworkModule(GenericNetworkModule):
	def __init__(self, hyperParams = None,modelType='SEP',featureSizePerProtein=1164):
		GenericNetworkModule.__init__(self,hyperParams)
		self.scaler = self.hyperParams.get('scaler',StandardScaler)
		self.modelType = self.hyperParams.get('modelType',modelType).upper()
		self.featureSize = self.hyperParams.get('featureSize',featureSizePerProtein)
		
	def genModel(self):
		if self.modelType != 'COMB':
			self.model = Du2017Model(self.hyperParams,self.modelType,self.featureSize)
		else:
			self.model = Du2017Model(self.hyperParams,self.modelType,self.featureSize*2)
		
	def loadFeatureData(self,featureFolder):
		featLst = []
		lst = ['AAC20','AAC400','DuMultiCTD_C','DuMultiCTD_D','DuMultiCTD_T','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','Grantham_Quasi_30','Schneider_Quasi_30','APSAAC30_2']
		for item in lst:
			featLst.append(featureFolder+item+'.tsv')
		featuresData = ProteinFeaturesHolder(featLst)
		self.dataLookup = {}
		for item in featuresData.rowLookup:
			self.dataLookup[str(item)] = featuresData.rowLookup[item]
		self.dataMatrix = torch.tensor(self.scaler().fit_transform(featuresData.data))
		#self.dataMatrix = torch.tensor(featuresData.data)
		self.featureSize = self.dataMatrix.shape[1]
		

	
	
class Du2017DeepNetworkModuleSep(Du2017DeepNetworkModule):
	def __init__(self, hyperParams = None,featureSizePerProtein=1164):
		super().__init__(hyperParams,'SEP',featureSizePerProtein)

class Du2017DeepNetworkModuleComb(Du2017DeepNetworkModule):
	def __init__(self, hyperParams = None,featureSizePerProtein=1164):
		super().__init__(hyperParams,'COMB',featureSizePerProtein)

