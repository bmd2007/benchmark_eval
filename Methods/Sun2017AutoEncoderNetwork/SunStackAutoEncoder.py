#Based on paper Sequence-based prediction of protein protein interaction using a deep-learning algorithm by Sun, Zhou, Lai, and Pei
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
from GenericNetworkModel import GenericNetworkModel
import PPIPUtils
import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
import torch
from NetworkRunner import NetworkRunner
from NetworkRunnerAuto import NetworkRunnerAuto
from SimpleDataset import SimpleDataset
import torch
import torch.nn as nn
#note, may come back and use dictionary dataset later to reduce memory

class SunNetworkAC(nn.Module):
	def __init__(self,inputSize=210,hiddenSize=400,seed=1):
		super(SunNetworkAC, self).__init__()
		torch.manual_seed(seed)
		self.act = nn.ReLU()
		self.layer0 = nn.Linear(inputSize*2,hiddenSize)
		self.layerD0 = nn.Linear(hiddenSize,inputSize*2)
		self.layerO0 = nn.Linear(hiddenSize,2)
		self.mode = 0
		
	def forward(self,x):
		if self.mode == 0:
			return self.forwardAuto(x)
		elif self.mode == 1:
			return self.forwardReg(x)
	
	def forwardAuto(self,x):
		x = self.layer0(x)
		x = self.act(x)
		x = self.layerD0(x)
		return x
	
	def forwardReg(self,x):
		x = self.layer0(x)
		x = self.act(x)
		x = self.layerO0(x)
		return x


	

#hyperparams:
#fullGPU - transfer full training data onto gpu
#deviceType - cpu or cuda
#paper suggests using an lr=1, and momentum of 0.5, using sgd.  Those numbers seem a little high, so I'm leaving the scheduler in place.  
#I don't see a number of epochs suggested either, so I'm using a scheduler
#I'm setting the scheduler to start decreasing at a high threshold, as this network tends to incremently decrease the loss for a long time without any real improvement
#the accuracy doesn't change much on 10 fold cross validation from doing this 
#(seems to fluctuate +/-.5% by stopping early, running full 200 iters scored 0.04% better on 10-fold avg Pan's data, which is a small amount)
#for now, I'm just doing the training process again using the softmax and the same learning rate
class SunStackedAutoModel(GenericNetworkModel):
	def __init__(self,hyp={},inputSize=210,hiddenSize=400,fullGPU=False,deviceType=None,numEpochs=200,batchSize=256,lr=1,momentum=0.5,schedFactor=0.5,schedThresh=3e-2,minLr=1e-2,optType='SGD'):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,momentum=momentum,schedFactor=schedFactor,schedThresh=schedThresh,minLr=minLr,optType=optType)
		
		self.inputSize=  inputSize #based on size of data, cannot be override through hyp
		self.hiddenSize = hyp.get('hiddenSize',hiddenSize)
		
	def genModel(self):
		self.net = SunNetworkAC(self.inputSize,self.hiddenSize,self.seed)
		self.modelAuto = NetworkRunnerAuto(self.net,hyp=self.hyp)
		self.model = NetworkRunner(self.net,hyp=self.hyp)
		
	#train network
	def fit(self,features,classes):
		self.genModel()
		self.net.mode=0
		self.modelAuto.trainNumpy(features,self.numEpochs,self.seed,full_gpu=self.fullGPU,min_lr=self.minLr)
		self.net.mode=1
		self.model.trainNumpy(features,classes,self.numEpochs,self.seed,full_gpu=self.fullGPU,min_lr=self.minLr)
		
	#predict on network
	def predict_proba(self,features):
		probs,loss = self.model.predictNumpy(features)
		return probs
		

class SunStackAutoEncoder(GenericModule):
	def __init__(self, hyperParams = None, inputSize=210,hiddenSize=400):
		GenericModule.__init__(self,hyperParams)
		self.featDict = self.hyperParams.get('featDict',{'all':['AC30.tsv']})
		self.inputSize=self.hyperParams.get('inputSize',inputSize)
		self.hiddenSize = self.hyperParams.get('hiddenSize',hiddenSize)
		
	def genModel(self):
		self.model = SunStackedAutoModel(self.hyperParams,self.inputSize,self.hiddenSize)

	def loadFeatureData(self,featureFolder):
		super().loadFeatureData(featureFolder)
		self.inputSize = self.featuresData['all'].data.shape[1]
		
	
class SunStackAutoEncoderAC(SunStackAutoEncoder):
	def __init__(self,hyperParams):
		super().__init__(hyperParams)
		
class SunStackAutoEncoderCT(SunStackAutoEncoder):
	def __init__(self,hyperParams):
		hyperParams['featDict'] = {'all':['ConjointTriad.tsv']}
		super().__init__(hyperParams,343,400)
