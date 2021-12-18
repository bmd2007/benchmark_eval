#Based on the paper Predicting protein-protein interactions through sequence-based deep learning Hashemifar, Neyshabur, Khan, and Xu
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


class NetworkHashemifar(nn.Module):
	def __init__(self,featureSize=20,proteinLength=512,num_layers=4,seed=1,deviceType=None):
		super(NetworkHashemifar, self).__init__()
		
		if deviceType is None or deviceType not in ['cpu','cuda']:
			self.deviceType = 'cuda' if torch.cuda.is_available() else 'cpu'

		torch.manual_seed(seed)
		
		self.activation = nn.ReLU()
		
		self.convLst = nn.ModuleList()
		self.batchNormLst = nn.ModuleList()
		self.poolLst = nn.ModuleList()

		curProtLength = proteinLength
		minSize = proteinLength//(2**(num_layers-1))
		for i in range(0,num_layers):
			self.convLst.append(nn.Conv1d(featureSize if i == 0 else minSize*(2**(i-1)),minSize*(2**i),5,padding=2))
			#transpose
			self.batchNormLst.append(nn.BatchNorm1d(minSize*(2**i)))
			if i < num_layers-1:
				self.poolLst.append(nn.AvgPool1d(4))
				curProtLength = curProtLength//4
			else:
				self.poolLst.append(nn.AvgPool1d(curProtLength))
				curProtLength=1
			#tranpose

		outputSize = minSize*(2**(num_layers-1))
		#x and x2 are not saved, just grabbing their bias and weights
		x = torch.nn.Linear(outputSize,outputSize).to(self.deviceType)
		x2 = torch.nn.Linear(outputSize,outputSize).to(self.deviceType)
		self.weight_init(x)
		self.weight_init(x2)
		
		#detach from network, so they don't require a gradient
		self.W1 = x.weight.detach()
		self.B1 = x.bias.detach()
		self.W2 = x2.weight.detach()
		self.B2 = x2.bias.detach()
		
		
		self.randomBatch1 = nn.BatchNorm1d(outputSize*2)
		self.randomBatch2 = nn.BatchNorm1d(outputSize*2)
		
		self.finalLayer = nn.Linear(outputSize*2,2)
		
		
		self.apply(self.weight_init)
		
	def weight_init(self,layer):
		if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv1d):
			torch.nn.init.normal_(layer.weight)
		if isinstance(layer,nn.LSTM):
			for i in range(0,layer.num_layers):
				torch.nn.init.normal_(layer._parameters['weight_ih_l'+str(i)])
				torch.nn.init.normal_(layer._parameters['weight_hh_l'+str(i)])

		
	def forward(self,x):
		(protA, protB) = x
		
		protLst = []
		for item in [protA,protB]:
			item = item.permute(0,2,1) #move channels (features per amino acid) from axis 2 to axis 1
			for i in range(0,len(self.convLst)):
				item = self.convLst[i](item)
				item = self.batchNormLst[i](item)
				item = self.activation(item)
				item = self.poolLst[i](item)
				
			item = item.squeeze(2) #remove extra dimension
			protLst.append(item)
			
		
		#each protein is now batchSize x outputSize
		protA = protLst[0]
		protB = protLst[1]
		
		#random projection on untrained linear layers, creating proteins sized batchSize x outputSize*2
		protA = torch.cat(((protA @ self.W1 + self.B1),(protA @ self.W2 + self.B2)),dim=1)
		protB = torch.cat(((protB @ self.W2 + self.B2),(protB @ self.W1 + self.B1)),dim=1)
		
		#finish random projection with individual batch norms and activations
		protA = self.randomBatch1(protA)
		protA = self.activation(protA)
		protB = self.randomBatch2(protB)
		protB = self.activation(protB)
		
		x = protA * protB
		x = self.finalLayer(x)
		
		return x

	
				

class Hashemifar2018Model(GenericNetworkModel):
	def __init__(self,hyp={},featureSize=20,maxProteinLength=512,numLayers=4,fullGPU=False,deviceType=None,numEpochs=100,batchSize=100,lr=1e-2,momentum=0.9,minLr=2e-4,schedFactor=.4,schedPatience=3,schedThresh=1e-2,weightDecay=1e-2,optType='SGD'):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,momentum=momentum,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,weightDecay=weightDecay,optType=optType)
	
		
		self.maxProteinLength = maxProteinLength #cannot be overriden, passed in from module, based on protein slicing
		self.numLayers = hyp.get('numLayers',numLayers)
		self.featureSize = featureSize #can not override through hyperparams, passed in from modeul, based on provided encoding files
		
	def genModel(self):
		self.net = NetworkHashemifar(self.featureSize,self.maxProteinLength,self.numLayers,deviceType=self.deviceType)
		self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
		
	#train network
	def fit(self,pairLst,classes,dataMatrix):
		self.genModel()
		#batch norm fails on batch size 1
		#We provide this fix here since the number of data points in the training data is not user controlled
		#ie We create an artifical number of training points based on the length of the proteins in the training pairs
		#In other networks, the user should fix the data or batch size if they have problems
		if pairLst.shape[0] % self.model.batch_size == 1: 
			pairLst = pairLst[:-1,:]
			classes = classes[:-1]
		dataset = SimpleTorchDictionaryDataset(dataMatrix,pairLst,classes,full_gpu=self.fullGPU)
		self.model.train(dataset, self.numEpochs,seed=self.seed,min_lr=self.minLr)
		
		
#Splits proteins into chunks of size at most NameError
#This means longer proteins get trained more in the network
#Also, due to max operator in predictions, longer proteins ore more likely to be predicted as interacting?

class Hashemifar2018DeepNetworkModule(GenericNetworkModule):
	def __init__(self, hyperParams = {}, proteinSlice=256, featureSize=20, seed=1):
		GenericNetworkModule.__init__(self,hyperParams)
		#slice size, = 1/2 window size of sequences for protein data
		self.proteinSlice = self.hyperParams.get('proteinSlice',proteinSlice)
		#number of features per pssm, default is 20, 1 per standard amino acid.  This value is overwritten after loading the feature data automatically
		self.featureSize = self.hyperParams.get('featureSize',featureSize)
		self.seed = self.hyperParams.get('seed',seed)
		self.hyperParams['seed'] = self.seed
		
	def genModel(self):
		self.model = Hashemifar2018Model(self.hyperParams,self.featureSize,self.proteinSlice*2)

	def loadFeatureData(self,featureFolder):
		tup = load(featureFolder+'PSSMLst.pssms')
		names, pssms = tup
		self.dataLookup = {}
		self.dataMatrix = []
		#split up pssms into chunks of at most length = self.proteinSlice*2
		
		for i in range(0,len(names)):
			#create a list for each protein, mapping to all of its chunks
			self.dataLookup[names[i]] = []
			stLen = pssms[i].shape[0]
			letters = pssms[i].shape[1]
			
			#small protein, fits in single chunk
			if stLen <= self.proteinSlice*2:
				#add index to list
				self.dataLookup[names[i]].append(len(self.dataMatrix))
				#copy data into tensor
				t = torch.zeros((self.proteinSlice*2,letters))
				t[0:stLen,:] = torch.tensor(pssms[i])
				self.dataMatrix.append(t.unsqueeze(0)) #add empty dim 0 for stacking
			else:
				#multi chunk protein, split into overlapping windows
				for j in range(0,stLen-self.proteinSlice,self.proteinSlice):
					#add index to list
					self.dataLookup[names[i]].append(len(self.dataMatrix))
					#get start and stop for chunk
					startIdx = j
					stopIdx = min(stLen,startIdx+2*self.proteinSlice)
					#copy data into tensor
					t = torch.zeros((self.proteinSlice*2,letters))
					t[0:(stopIdx-startIdx),:] = torch.tensor(pssms[i][startIdx:stopIdx,:])
					self.dataMatrix.append(t.unsqueeze(0)) #add empty dim 0 for stacking
					
		self.dataMatrix = torch.vstack(self.dataMatrix).float()
		self.featureSize = self.dataMatrix.shape[2]

	#convert all pairs to sets of indices for training, using dictionary mapping each protein to a list of dataMatrix IDs
	def convertFeaturesToIndices(self, pairs, classes=None):
		pairs = pairs.tolist()
		if classes is not None:
			classes = classes.tolist()
		newPairs = []
		newClasses = []
		indexLst = []
		for i in range(0,len(pairs)):
			#for all combinations of IDs mapping to proteins A and B, making a training pair
			#also record the pair's index to recombine predictions later
			for item1 in self.dataLookup[str(pairs[i][0])]:
				for item2 in self.dataLookup[str(pairs[i][1])]:
					newPairs.append((item1,item2))
					if classes is not None:
						newClasses.append(classes[i])
					indexLst.append(i)
		return np.asarray(indexLst), np.asarray(newPairs), (np.asarray(newClasses) if classes is not None else None)
		
	def convertPredictionsToOrgIndices(self,pairIndexLst,preds):
		numPredictions = np.max(pairIndexLst)+1
		finalPreds = np.zeros((numPredictions,2))
		for i in range(0,preds.shape[0]):
			idx = pairIndexLst[i]
			#if prediction in positive class greater than current best predict, take it as the new prediction (max aggregation)
			if finalPreds[idx][1] < preds[i][1]:
				finalPreds[idx] = preds[i]
		return finalPreds
		
	#when running fit or predict, pass in the dataMatrix
	def fit(self,trainFeatures,trainClasses,model=None):
		pairIndexLst, featureIdxLst, classLst = self.convertFeaturesToIndices(trainFeatures,trainClasses)
		if model is not None:
			model.fit(featureIdxLst,classLst,self.dataMatrix)
		else:
			self.genModel() #create a new model from scratch, ensuring we don't overwrite the previously trained one
			self.model.fit(featureIdxLst,classLst,self.dataMatrix)

	#swap out pair data for their indices in the data matrix, and return them as features
	def genFeatureData(self,pairs,dataType='Train'):	
		classData = np.asarray(pairs[:,2],dtype=np.int32)
		featsData = pairs[:,0:2]
		#do nothing, will handle in predict and fit functions
		return featsData, classData

	#when running fit or predict, pass in the dataMatrix
	def predict_proba(self,predictFeatures,predictClasses,model=None):
		pairIndexLst, featureIdxLst, classLst = self.convertFeaturesToIndices(predictFeatures)
		if model is not None:
			preds = model.predict_proba(featureIdxLst,self.dataMatrix)
		else:
			preds = self.model.predict_proba(featureIdxLst,self.dataMatrix)
		finalPreds = self.convertPredictionsToOrgIndices(pairIndexLst,preds)
		return (finalPreds,predictClasses)
