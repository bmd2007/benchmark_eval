#Based on skip-gram model
#Distributed Representations of Words and Phrases and their Compositionality
#Tomas Mikolov Ilya Sutskever Kai Chen Greg Corrado Jeffrey Dean

#Note, this model is built based on the concept of a small corpus size, usually around 20 for amino acids
#Some things in this file (particularly our implementation of the loss function) may not scale well to larger corpuses
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import numpy as np
#add parent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import PPIPUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump, load
from Methods.SimpleDataset import SimpleDataset
from Methods.NetworkRunner import NetworkRunner
from AACounter import AACounter
from torch.utils import data
from joblib import dump, load
import math

class SkipGramDatasetTrain(data.Dataset):
	def __init__(self,torchVectors,windowSize,preCompute=False,full_gpu=False):
		self.torchVectors=torchVectors
		self.windowSize=windowSize
		self.indices = []
		idx = 0
		for item in self.torchVectors:
			a = torch.arange(0,len(item)-windowSize*2-1)
			b = torch.zeros(a.shape)+idx
			self.indices.append(torch.cat((a.unsqueeze(0),b.unsqueeze(0)),dim=0).T)
			idx += 1
		self.indices = torch.cat(self.indices)
		
		self.full_gpu = full_gpu
		self.preCompute = preCompute
		if preCompute:
			self.fullX = []
			self.fullY = []
			for item in self.torchVectors:
				letters = item.unsqueeze(0).T
				newLetters = letters[0:letters.shape[0]-windowSize]
				for i in range(1,windowSize):
					newLetters = torch.cat((newLetters,letters[i:letters.shape[0]-windowSize+i]),dim=1)
				newLetters = torch.cat((newLetters[0:newLetters.shape[0]-windowSize-1],newLetters[windowSize+1:]),dim=1)
				self.fullX.append(item[windowSize:letters.shape[0]-windowSize-1])
				self.fullY.append(newLetters)
				
			self.fullX = torch.cat(self.fullX)
			self.fullY = torch.cat(self.fullY)
		
			
	def activate(self):
		if self.full_gpu: #push everything to gpu
			if self.preCompute:
				self.fullX = self.fullX.cuda()
				self.fullY = self.fullY.cuda()
			else:
				for i in range(0,len(self.torchVectors)):
					self.torchVectors[i] = self.torchVectors[i].cuda().long()
				self.indices = self.indices.cuda().long()
	
	def deactivate(self):
		if self.full_gpu: #push everything to cpu
			if self.preCompute:
				self.fullX = self.fullX.cpu()
				self.fullY = self.fullY.cpu()
			else:
				for i in range(0,len(self.torchVectors)):
					self.torchVectors[i] = self.torchVectors[i].cpu().long()
				self.indices = self.indices.cpu().long()
		
	def __len__(self):
		return self.indices.shape[0]
		
	def __getitem__(self,index):
		if self.preCompute:
			xOrg = self.fullX[index].unsqueeze(0)
			yOrg = self.fullY[index]
		
		#this runs slow, so preCompute is recommended.
		#It probably runs slower due to the concatenation of 2 vectors for the y data.  May be quicker to pass entire range and recode network to recognize middle value as input
		#could also be slower due to different length vectors containing raw data (torchVectors), rather than a single matrix.  More experimentation is necessary to determine the reason.
		else:
			col,row = self.indices[index,:]
			col+=self.windowSize
			
			#get input data (single letter/value)
			xOrg = self.torchVectors[row][col].unsqueeze(0)
			
			#get classData, window around xOrg
			yOrg = torch.cat((self.torchVectors[row][col-self.windowSize:col],self.torchVectors[row][col+1:col+self.windowSize+1]))
			
		#for lazyness, we are going to hack the data so it works without changing the train function of network runner (so I don't have to copy 10-30 lines of code)
		#to do that, all data needed by the network needs to be in x, and the loss function should only need the values from (net(x),y).
		#to do that, we will return concat(x,y) as train data, and y as class data
		return (torch.cat((xOrg,yOrg)),yOrg)
		
class SkipGramNet(nn.Module):
	def __init__(self,corpusSize=20,hiddenSize=300,negativeSize=5,corpusSmall=True,deviceType='cpu',seed=1):
		super().__init__()
		torch.manual_seed(seed)
		self.corpusSize=corpusSize
		self.hiddenSize=hiddenSize
		self.negativeSize=negativeSize
		self.corpusSmall=corpusSmall
		self.deviceType=deviceType
		self.layer0 = nn.Embedding(corpusSize,hiddenSize,sparse=True) #sparse gradient
		#layer1rev is the reverse of what would normally be layer 1, as we are going to 
		self.layer1rev = nn.Embedding(corpusSize,hiddenSize,sparse=True) #sparse gradient
		
	def forward(self,x):
		
		#using our dataset hack, the first value in each row of batch x is the word were are training on
		#and the remaining values are the positive examples
		y = x[:,1:] 
		x = x[:,0]
		
		
		#embed x in layer 0
		x = self.layer0(x)
		x = x.unsqueeze(1)
		#calcuate weights of positive classes from back of model to middle
		newY = self.layer1rev(y)
		
		#multiple x vals from layer 1 by positive data values from layer 2 to get positive loss
		pos = torch.sum(torch.mul(x,newY),dim=2)
		
		#do negative sampling
		#two ways.  We could generate random numbers for each non-positive class for each batch sample (this creates unique values)
		#or we could generate random numbers, assuming the odds of accidentally choosing a positive class are lower (which allows duplicate values)
			
		#way 1, random for each non-positive.  Good for small corpus sizes (such as the 20 amino acids)
		if self.corpusSmall:
			#gen random numbers for all corpus words, for all batchs
			n = torch.rand((y.shape[0],self.corpusSize),device=self.deviceType)
			#zero out positive values
			#create row indices for positive values, matrix of size y where each row's values is the row index
			row_idx = torch.arange(0,y.shape[0],device=self.deviceType).unsqueeze(0).T.tile(1,y.shape[1])
			#zero out values
			n[row_idx,y] = 0
			#get top k values, where k is negativeSize, per row, as negative indices for batch
			neg = torch.topk(n,self.negativeSize,dim=1)[1]
				
		#way 2, gen random numbers, hope they don't match the positive set.
		else:
			neg = torch.randint(0,self.corpusSize,(y.shape[0],self.negativeSize))
				
		#calculate values for negative data
		neg = self.layer1rev(neg)
		neg = torch.sum(torch.mul(x,neg),dim=2)
			
		#concate and return
		retData = torch.cat((pos,neg),dim=1)
			
		return retData
		
class SkipGramNetworkRunner(NetworkRunner):
	def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='SGD',weight_decay=0,sched_factor=None,sched_patience=None,sched_cooldown=None,sched_thresh=None,predictSoftmax=True):
		super().__init__(net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax)
		self.lgs = nn.LogSigmoid()
		
	#continuing our hack, output returned (from skipgramnet) a concatenatation of p positive values, and n negative values
	#x, the class data, contains p integers
	#we can use the size of x to split output into positive and negative data
	def getLoss(self,output,x):
		#split outputs into positive values and negative values
		
		#get number of positive values, by measuring the size of the class variable
		posNums = x.shape[1]
		
		posVals = output[:,:posNums]
		negVals = output[:,posNums:]
		
		
		#passed in no negative data, which can happen during predict, if requesting probabilities for all values.  Just return zero loss
		if posNums == output.shape[1]:
			loss = torch.mean(posVals*0.0)
			
		else:
			#calculate loss per entry
			posVals = self.lgs(posVals)
			negVals = self.lgs(-negVals)
			loss = torch.mean(posVals.sum(1)+negVals.sum(1)) * -1
		return loss
	
	

#1 epoch is more than enough to train a network this small with enough proteins
def SkipGram(fastas, fileName, windowSize=7, negativeSize=5, hiddenSize=7, corpusSmall=True, numEpochs=1,groupings=None,groupLen=1,sorting=False, flip=False,excludeSame=False, preTrained=False, deviceType='cpu',fullGPU=False,saveModel=True,preCompute=True,softMax=False):
	
	if groupings is not None:
		groupMap = {}
		idx = 0
		for item in groupings:
			for let in item:
				groupMap[let] = idx
			idx += 1
		numgroups = len(groupings)
	else:
		groupMap = None
		numgroups=20

	parsedData = AACounter(fastas, groupMap, groupLen, sorting=sorting,flip=flip,excludeSame=excludeSame,getRawValues=True,deviceType=deviceType)
	
	#number of unique groups, typically 20 amino acids, times length of our embeddings, typically 1, equals the corpus size
	corpusSize = numgroups**groupLen
	
	
	skipModel = SkipGramNet(corpusSize,hiddenSize,negativeSize,corpusSmall,deviceType)

	skipRunner = SkipGramNetworkRunner(skipModel)
	if preTrained is True and os.path.isfile(fileName.split('.')[0]+'_skipgram.model'):
		skipRunner.load(fileName.split('.')[0]+'_skipgram.model')
	elif preTrained is not None and preTrained is not False:
		skipRunner.load(preTrained)
	else: #do training
		skipTrainData = SkipGramDatasetTrain([x[1] for x in parsedData[1:]],windowSize,preCompute,fullGPU)
		skipRunner.train(skipTrainData,numEpochs)
		if saveModel is True:
			skipRunner.save(fileName.split('.')[0]+'_skipgram.model')
		elif saveModel is not None and saveModel is not False:
			skipRunner.save(saveModel)
		
	
	#get values for each letter as prediction probabilities
	#each row will represent 1 letter, and be the length of hiddenSize
	#hidden layer is one-hot encoding times first layer, which is just the weights of the first layer
	vals = skipModel.layer0.weight.detach()
	if softMax:
		vals = F.softmax(vals,1)
	


	f = open(fileName,'w')
	#create Matrix
	#creating a matrix obviously isn't necessary for this method, but doing this allows us to avoid creating another type of feature file to parse
	#use i+1 instead of i, so, if algorithm zero pads, it won't match any value
	for i in range(0,corpusSize):
		lst = vals[i,:].tolist()
		f.write(','.join(str(k) for k in lst)+'\n')
	f.write('\n')

	for item in parsedData[1:]:
		name = item[0]
		stVals = item[1].cpu().numpy()
		f.write(name+'\t'+','.join(str(k) for k in stVals) +'\n')
	f.close()
	
	return None
