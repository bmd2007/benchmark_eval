#Based on paper Protein Interaction Network Reconstruction Through Ensemble Deep Learning With Attention Mechanism by Li, Zhu, Ling, and Liu

import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from GenericModule import GenericModule

import PPIPUtils
import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
import torch
from LiModels import *
from NetworkRunner import NetworkRunner
from SimpleDataset import SimpleDataset
from torch.utils import data as torchData



#hyperparams:
#fullGPU - transfer full training data onto gpu
#deviceType - cpu or cuda
#k - number of cross folder per network, default in paper is 5, but we set the default to 3 to make it a bit faster.
class LiDeepModel(object):
	def __init__(self,hyp={},fullGPU=False,deviceType=None,k=3,numEpochs=200,layer1Lr=2e-1,layer2Lr=1e-1,batchSize=256, minLr=1e-4, schedFactor=.1,schedPatience=2,schedThresh=5e-2):
	
		self.hyp = hyp
		self.fullGPU = hyp.get('fullGPU',fullGPU)
		self.numEpochs = hyp.get('numEpochs',numEpochs)
		self.seed = hyp.get('seed',1)
		self.minLr = hyp.get('minLr',minLr)
		self.layer1Lr = hyp.get('layer1Lr',layer1Lr)
		self.layer2Lr = hyp.get('layer1Lr',layer2Lr)
		self.k = hyp.get('k',k)
		
		#move network runner properties into hyperparams list if needed
		hyp['batchSize'] = hyp.get('batchSize',batchSize)
		hyp['schedThresh'] = hyp.get('schedThresh',schedThresh)
		hyp['schedPatience'] = hyp.get('schedPatience',schedPatience)
		hyp['schedFactor'] = hyp.get('schedFactor',schedFactor)
		hyp['deviceType'] = hyp.get('deviceType',deviceType)
		hyp['optType'] = hyp.get('optType','SGD')
		hyp['seed'] = self.seed
				
		
		self.model = None
	
		#create all the networks.  Default is 49 networks (16 networks, 3 fold cross)
		
		#paper doesn't specific how to vary networks
		#currently, we can vary the number of convolution layers, the number of heads in the multi-attention model, and the number of linear layers
		#conv_size, the size of the convolution (size x size convolution layer)
		#also, we could modify the kernal size of the convolution layer, poolSize of the the pooling layers, and the random seed
		
		#embed_dim (conv_size) must be divisible by num_heads
		
		#init funciton for deep model
		#def __init__(self,vector_size,conv_size,conv_repeats,kernel_size,num_heads, poolSize, numLinLayers,seed=1):
		
		self.models ={'AAC':{},'LD':{},'CT':{},'PAAC':{}}
		for i in range(0,k):
			self.models['AAC'][i] = [
			NetworkRunner(ModelLiDeep(210,24,1,3,4,10,3,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(210,24,1,3,3,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(210,24,1,3,4,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(210,24,2,3,3,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp)
			]
						
			self.models['LD'][i] = [
			NetworkRunner(ModelLiDeep(630,24,1,3,4,10,3,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(630,24,1,3,3,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(630,24,1,3,4,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(630,24,2,3,3,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp)
			]
			
			self.models['CT'][i] = [
			NetworkRunner(ModelLiDeep(343,24,1,3,4,10,3,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(343,24,1,3,3,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(343,24,1,3,4,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(343,24,2,3,3,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp)
			]
			
			self.models['PAAC'][i] = [
			NetworkRunner(ModelLiDeep(35,24,1,3,4,10,3,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(35,24,1,3,3,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(35,24,1,3,4,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp),
			NetworkRunner(ModelLiDeep(35,24,2,3,3,10,2,self.seed),lr=self.layer1Lr,hyp=self.hyp)
			]
		
		self.models['Ensemble'] = NetworkRunner(EnsembleNetwork(self.seed),lr=self.layer2Lr,hyp=self.hyp)
		
		
		
			
		
	#save all network to files
	def saveAll(self,folderName):
		PPIPUtils.makeDir(folderName)
		for item in ['AAC','LD','CT','PAAC']:
			for i in range(0,self.k):
				for j in range(0,4):
					self.saveModel(item,i,j,folderName)
		self.saveModel('Ensemble',0,0,folderName)

	def saveModel(self,item,idx,idx2,folderName):
		PPIPUtils.makeDir(folderName)
		if item == 'Ensemble':
			self.models['Ensemble'].save(folderName+'Ensemble.out')
		else:
			self.models[item][idx][idx2].save(folderName+item+'_'+str(idx)+'_'+str(idx2)+'.out')
			
	#load all networks from files
	def loadAll(self,folderName):
		for item in ['AAC','LD','CT','PAAC']:
			for i in range(0,self.k):
				for j in range(0,4):
					self.loadModel(item,i,j,folderName)
		self.loadModel('Ensemble',0,0,folderName)

	def loadModel(self,item,idx,idx2,folderName):
		if item == 'Ensemble':
			name = folderName+'Ensemble.out'
		else:
			name = folderName+item+'_'+str(idx)+'_'+str(idx2)+'.out'
		try:
			if item == 'Ensemble':
				self.models[item].load(name)
			else:
				self.models[item][idx][idx2].load(name)
		except Exception as E:
			print('Error, cannot find file for',item,idx,idx2,'at location',name)
			print(E)
			exit(42)
		
		
	#create kfolds for training layers 1 and 2
	def createKFoldIdx(self,featLen,classes,k=5,seed=1):
		posTot = np.sum(classes)
		negTot = featLen-posTot
		allIdxs = np.arange(0,featLen)
		posIdx = allIdxs[classes==1]
		negIdx = allIdxs[classes==0]
		np.random.seed(seed)
		np.random.shuffle(posIdx)
		np.random.shuffle(negIdx)
		splitsPos = []
		splitsNeg = []
		for i in range(0,k):
			splitsPos.append((posIdx.shape[0]*i)//k)
			splitsNeg.append((negIdx.shape[0]*i)//k)
		splitsPos.append(posIdx.shape[0])
		splitsNeg.append(negIdx.shape[0])
		folds = []
		for i in range(0,k):
			test = np.hstack((posIdx[splitsPos[i]:(splitsPos[i+1])],negIdx[splitsNeg[i]:(splitsNeg[i+1])]))
			train = np.hstack((posIdx[:splitsPos[i]],posIdx[(splitsPos[i+1]):],negIdx[:splitsNeg[i]],negIdx[(splitsNeg[i+1]):]))
			folds.append((train,test))
		
		return folds

	#train each copy of the 4*4*k networks on layer 1 on k-1 folds, use kth fold for validation, and use 4*4*1 outputs per held out pair to train layer 2
	def fit(self,features,classes):
		layer2Features = []
		modelIdx=0
		folds = self.createKFoldIdx(features['AAC'].shape[0],classes,self.k,self.seed)
		for netType in ['AAC','LD','CT','PAAC']:
			for netIdx in range(0,len(self.models[netType][0])):
				curValFeatures = np.zeros(features['AAC'].shape[0])
				for foldIdx in range(0,self.k):
					print('modelIdx',modelIdx,foldIdx,netIdx,netType)
					modelIdx+=1
					trainIdx = folds[foldIdx][0]
					testIdx = folds[foldIdx][1]
					trainFeats = features[netType][trainIdx,:]
					testFeats = features[netType][testIdx,:]
					trainClasses = classes[trainIdx]
					testClasses = classes[testIdx]
					self.models[netType][foldIdx][netIdx].trainNumpy(trainFeats,trainClasses,self.numEpochs,self.seed,min_lr=self.minLr,full_gpu=self.fullGPU)
					probs, loss = self.models[netType][foldIdx][netIdx].predictNumpy(testFeats,testClasses)
					
					#classPredict = np.argmax(probs,1)
					curValFeatures[testIdx] = probs[:,1]
						
					
				layer2Features.append(np.expand_dims(curValFeatures,1))
					
		layer2Features = np.hstack(layer2Features)
		
		self.models['Ensemble'].trainNumpy(layer2Features,classes,self.numEpochs,self.seed,min_lr=self.minLr,full_gpu=self.fullGPU)
		
	#just to clarify, my understanding of the code is that in the training loop, after layer 1, each value is discretized to 0 or 1 for the class
	#this also happens testing, but, since testing isn't split randomly over k folders, test values are averaged over k networks, producted a number in the range 0, 1/k, 2/k, . . . 1
	#which means that the test data will be floating point entering layer 2, but layer 2 has only been trained on binary data?
	def predict_proba(self,features):
		layer2Features = []
		features['PAAC'] = features['PAAC'] * 100
		for netType in ['AAC','LD','CT','PAAC']:
			predictDataset = SimpleDataset(features[netType],None)
			predictLoader = torchData.DataLoader(predictDataset,**self.models[netType][0][0].getLoaderArgs(False,True))
			for netIdx in range(0,len(self.models[netType][0])):
				curValFeatures = np.zeros(len(predictDataset))
				for foldIdx in range(0,self.k):
					probs, loss = self.models[netType][foldIdx][netIdx].predictFromLoader(predictLoader)
					#classPredict=np.argmax(probs,1)
					curValFeatures += probs[:,1] #classPredict
				curValFeatures = curValFeatures / self.k
				layer2Features.append(np.expand_dims(curValFeatures,1))

		layer2Features = np.hstack(layer2Features)
		probs, loss = self.models['Ensemble'].predictNumpy(layer2Features,None)
		return probs
		

class LiDeepNetworkModule(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		
	def genModel(self):
		self.model = LiDeepModel(self.hyperParams)
		

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
		if fname[-1] == '/':
			self.saveModelToFolder(fname)
		else:
			print('Error, unable to save multiple models to single file')
			exit(42)
		
	def loadModelFromFile(self,fname):
		if fname[-1] == '/':
			self.loadModelFromFolder(fname)
		else:
			print('Error, unable to load multiple models to single file')
			exit(42)

		
	def loadFeatureData(self,featureFolder):
		self.featuresData = {}
		print('loading')
		self.featuresData['AAC'] = ProteinFeaturesHolder([featureFolder + 'AC30.tsv'])
		self.featuresData['PAAC'] = ProteinFeaturesHolder([featureFolder + 'PSAAC15.tsv'])
		self.featuresData['CT'] = ProteinFeaturesHolder([featureFolder + 'ConjointTriad.tsv'])
		self.featuresData['LD'] = ProteinFeaturesHolder([featureFolder + 'LD10_CTD_ConjointTriad_C.tsv',featureFolder + 'LD10_CTD_ConjointTriad_T.tsv',featureFolder + 'LD10_CTD_ConjointTriad_D.tsv'])
		print('done loading')
		
	def genFeatureData(self,pairs,dataType='Train'):
		classData = pairs[:,2]
		featData = {}
		print('generating')
		featData['AAC'] = self.featuresData['AAC'].genData(pairs)
		featData['PAAC'] = self.featuresData['PAAC'].genData(pairs)
		featData['CT'] = self.featuresData['CT'].genData(pairs)
		featData['LD'] = self.featuresData['LD'].genData(pairs)
		print('done generating')
		return featData, classData
	