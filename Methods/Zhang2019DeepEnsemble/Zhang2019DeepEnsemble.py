#Based on Protein–protein interactions prediction based on ensemble deep neural networks by Zhang, Yu, Xia, and Wang
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

import torch.nn as nn
import PPIPUtils
import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
import torch
from NetworkRunnerCollate import NetworkRunnerCollate
from SimpleDataset import SimpleDataset
from torch.utils import data as torchData
from SimpleTorchDictionaryDataset import SimpleTorchDictionaryDataset




class ZhangNetwork(nn.Module):
	def __init__(self,parameterLength=420,firstLayer=512,reductionRate=4,dropout=0.5,ens=False,seed=1,deviceType=None):
		super(ZhangNetwork, self).__init__()
		
		if deviceType is None or deviceType not in ['cpu','cuda']:
			self.deviceType = 'cuda' if torch.cuda.is_available() else 'cpu'
		
		torch.manual_seed(seed)
		#true if ensemble layer
		self.ens = ens
		
		self.activation= nn.ReLU()
		self.drop = nn.Dropout(dropout)
		self.linearLst = nn.ModuleList()
		self.batchNormLst = nn.ModuleList()
		
		#create first layers
		self.linearLst.append(nn.Linear(parameterLength,firstLayer))
		self.batchNormLst.append(nn.BatchNorm1d(firstLayer))
		
		#continue creating layers until only 2 output nodes remain
		x = firstLayer
		while x > 2 * reductionRate:
			x2 = x//reductionRate
			self.linearLst.append(nn.Linear(x,x2))
			self.batchNormLst.append(nn.BatchNorm1d(x2))
			x = x2
		
		#create final layer
		self.linearLst.append(nn.Linear(x,2))
		self.batchNormLst.append(nn.BatchNorm1d(2))
		
		self.apply(self.weight_init)
		
	def weight_init(self,layer):
		if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv1d):
			torch.nn.init.xavier_normal_(layer.weight)
		if isinstance(layer,nn.LSTM):
			for i in range(0,layer.num_layers):
				torch.nn.init.xavier_normal_(layer._parameters['weight_ih_l'+str(i)])
				torch.nn.init.xavier_normal_(layer._parameters['weight_hh_l'+str(i)])
		
	def forward(self,x):
		if not self.ens:
			(protA, protB) = x
			x = torch.cat((protA,protB),dim=1).to(self.deviceType)
		else:
			x = x[0].to(self.deviceType)
			
		for i in range(0,len(self.linearLst)):
			x = self.linearLst[i](x)
			x = self.batchNormLst[i](x)
			if i != len(self.linearLst)-1:
				x = self.activation(x)
				x = self.drop(x)
		return x



class ZhangNetworkRunner(NetworkRunnerCollate):
	def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=2,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=True,l1LossWeight=1e-5,hyp={}):
		NetworkRunnerCollate.__init__(self,net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax,hyp)
		self.l1LossWeight = hyp.get('l1LossWeight',l1LossWeight)

	#save memory by only keeping what we need, the average loss value per output
	def getLoss(self,output,classData):
		regLoss = super().getLoss(output,classData)
		l1Loss = self.getL1LossVal()
		return regLoss + l1Loss * self.l1LossWeight


#hyperparams:
#fullGPU - transfer full training data onto gpu
class ZhangDeepNetwork(GenericNetworkModel):
	def __init__(self,hyp={},fullGPU=False,deviceType=None,numEpochs=100,lrs={'AC':7e-4,'LD':1e-3,'MCD':3e-4,'ENS':1e-3},batchSize=256, minLrs={'AC':7e-4/32,'LD':1e-3/32,'MCD':3e-4/32,'ENS':1e-3/32}, schedFactor=.5,schedPatience=2,schedThresh=1e-2):

		GenericNetworkModel.__init__(self,hyp=hyp,numEpochs=numEpochs,batchSize=batchSize,schedFactor=schedFactor,schedPatience=schedPatience,schedThresh=schedThresh)
		
		self.minLrs = hyp.get('minLrs',minLrs)
		self.lrs = hyp.get('lrs',lrs)
		if 'lr' in hyp:
			del hyp['lr']#use values in lrs, don't override with lr hyperparam
		
		#create all 28 networks
		
		#paper doesn't specific exact configuration of networks, but does give suggestions per feature type
		#recommends larger depth for larger features (ranging from 2-9 layers)
		#widths ranging from 2**2 - 2**11,
		#and dropouts of 0.5 and 0.4 for AD, and 0.5 and 0.7 for other layers
		
		#within these groups, we are mostly just choosing randomly
		
		#network params:  parameterLength=420,firstLayer=512,reductionRate=4,dropout=0.5,seed=1,deviceType=None
		self.models ={'AC':[],'LD':[],'MCD':[],'ENS':[]}
		#9 AC models
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,256,8,0.5,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,128,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,32,4,0.4,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,256,8,0.4,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,4,2,0.4,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,16,2,0.5,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,128,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,256,8,0.4,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		self.models['AC'].append(ZhangNetworkRunner(ZhangNetwork(420,16,8,0.5,False,self.seed,self.deviceType),lr=self.lrs['AC'],hyp=self.hyp))
		
		#9 LD models
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,512,8,0.5,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,128,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,64,4,0.7,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,512,8,0.7,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,32,2,0.7,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,128,8,0.5,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,256,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,256,8,0.7,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		self.models['LD'].append(ZhangNetworkRunner(ZhangNetwork(1260,16,2,0.5,False,self.seed,self.deviceType),lr=self.lrs['LD'],hyp=self.hyp))
		
		#9 MCD models
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,512,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,128,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,2048,4,0.7,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,512,8,0.7,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,64,2,0.7,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,1024,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,256,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,512,2,0.7,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		self.models['MCD'].append(ZhangNetworkRunner(ZhangNetwork(3780,64,4,0.5,False,self.seed,self.deviceType),lr=self.lrs['MCD'],hyp=self.hyp))
		
		#1 final ensemble
		self.models['ENS'].append(ZhangNetworkRunner(ZhangNetwork(27,32,4,0.5,True,self.seed,self.deviceType),lr=self.lrs['ENS'],hyp=self.hyp))
		
		
		
			
	def saveModelToFile(self,fileName):
		self.saveAll(fileName)
		
	def loadModelFromFile(self,fileName):
		self.loadAll(fileName)
		
	#save all network to files
	def saveAll(self,folderName):
		PPIPUtils.makeDir(folderName)
		for item in ['AC','LD','MCD','ENS']:
			for i in range(0,len(self.models[item])):
				self.saveModel(item,i,folderName)

	def saveModel(self,item,idx,folderName):
		PPIPUtils.makeDir(folderName)
		self.models[item][idx].save(folderName+item+'_'+str(idx)+'.out')
			
	#load all networks from files
	def loadAll(self,folderName):
		for item in ['AC','LD','MCD','ENS']:
			for i in range(0,len(self.models[item])):
				self.loadModel(item,i,folderName)				
		
	def loadModel(self,item,idx,folderName):
		name = folderName+item+'_'+str(idx)+'.out'
		try:
			self.models[item][idx].load(name)
		except Exception as E:
			print('Error, cannot find file for',item,idx,'at location',name)
			print(E)
			exit(42)
		
		
	#ignores validation data 
	def fit(self,pairLst,classes,dataMatrix,validationPairs=None, validationClasses=None):
		layer2Features = []
		modelIdx=0
		finalProbs = np.zeros((pairLst.shape[0],len(self.models['AC'])+len(self.models['LD'])+len(self.models['MCD'])))
		for netType in ['AC','LD','MCD']:
			trainDataset = SimpleTorchDictionaryDataset(dataMatrix[netType],pairLst,classes,full_gpu=self.fullGPU)
			for netIdx in range(0,len(self.models[netType])):
				print('modelIdx',modelIdx,netIdx,netType)
				self.models[netType][netIdx].train(trainDataset,self.numEpochs,self.seed,min_lr=self.minLrs[netType])
				probs, loss = self.models[netType][netIdx].predict(trainDataset)
					
				finalProbs[:,modelIdx] = probs[:,1]
				modelIdx+=1
					
		print('modelIdx',modelIdx,0,'ENS')
		self.models['ENS'][0].trainNumpy(finalProbs,classes,self.numEpochs,self.seed,min_lr=self.minLrs['ENS'],full_gpu=self.fullGPU)
		
	#just to clarify, my understanding of the code is that in the training loop, after layer 1, each value is discretized to 0 or 1 for the class
	#this also happens testing, but, since testing isn't split randomly over k folders, test values are averaged over k networks, producted a number in the range 0, 1/k, 2/k, . . . 1
	#which means that the test data will be floating point entering layer 2, but layer 2 has only been trained on binary data?
	
	#predict on network
	def predict_proba(self,pairLst,dataMatrix):
		layer2Features = []
		finalProbs = np.zeros((pairLst.shape[0],len(self.models['AC'])+len(self.models['LD'])+len(self.models['MCD'])))
		modelIdx=0
		for netType in ['AC','LD','MCD']:
			predictDataset = SimpleTorchDictionaryDataset(dataMatrix[netType],pairLst,full_gpu=self.fullGPU)
			predictLoader = torchData.DataLoader(predictDataset,**self.models[netType][0].getLoaderArgs(False,False))
			for netIdx in range(0,len(self.models[netType])):
				probs, loss = self.models[netType][netIdx].predictFromLoader(predictLoader)
				finalProbs[:,modelIdx] = probs[:,1]
				modelIdx+=1
				
		probs, loss = self.models['ENS'][0].predictNumpy(finalProbs,None)
		return probs
		

class ZhangDeepModule(GenericNetworkModule):	
	def __init__(self, hyperParams = None):
		GenericNetworkModule.__init__(self,hyperParams)
		
	def genModel(self):
		self.model = ZhangDeepNetwork(self.hyperParams)
		

	def saveModelToFolder(self,folderName):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		self.model.saveAll(folderName)
		
	def loadModelFromFolder(self,folderName):
		if self.model is None:
			self.genModel()
		self.model.loadAll(folderName)

	
	def saveModelToFile(self,fname):
		if fname[-1] != '/':
			fname += '/'
		self.saveModelToFolder(fname)
		
	def loadModelFromFile(self,fname):
		if fname[-1] == '/':
			self.loadModelFromFolder(fname)
		else:
			print('Error, unable to load multiple models to single file')
			exit(42)

		
	def loadFeatureData(self,featureFolder):
		self.dataMatrix = {}
		print('loading')
		
		acFeats = ProteinFeaturesHolder([featureFolder + 'AC30.tsv'])
		self.dataMatrix['AC'] = torch.tensor(acFeats.data).float()
		self.dataLookup = acFeats.rowLookup
		
		ldFeats = ProteinFeaturesHolder([featureFolder + 'LD10_CTD_ConjointTriad_C.tsv',featureFolder + 'LD10_CTD_ConjointTriad_T.tsv',featureFolder + 'LD10_CTD_ConjointTriad_D.tsv'])
		self.dataMatrix['LD'] = torch.tensor(ldFeats.data).float()
		
		mcdFeats = ProteinFeaturesHolder([featureFolder + 'MCD5_CTD_ConjointTriad_C.tsv',featureFolder + 'MCD5_CTD_ConjointTriad_T.tsv',featureFolder + 'MCD5_CTD_ConjointTriad_D.tsv'])
		self.dataMatrix['MCD'] = torch.tensor(mcdFeats.data).float()
		
		print('done loading')
		
	#when running fit or predict, pass in the dataMatrix
	def fit(self,trainFeatures,trainClasses):
		self.validationRate =0 #this module currently does not handle validation data
		super().fit(trainFeatures,trainClasses)