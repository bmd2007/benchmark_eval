import numpy as np
import torch
from torch.utils import data
import SimpleTorchDictionaryDataset
import time

#uses a matrix of proteins, and indexes with a matrix of pairs in the form of (x0Idx, x1Idx, classVal)
class SimpleTorchDictionaryDataset(data.Dataset):
	def __init__(self,featureData,pairLst,classData=None,full_gpu=False,createNewTensor=False):
		if createNewTensor:
			self.data = torch.tensor(featureData).float()
		else:
			self.data=featureData
		
		#This makes no sense to me, but, it seems it is much faster to index into our data using 2 individual numbers return from numpy than using pytorch
		#Testing using invidual indexing (x0=data[index[0];x1=data[index[1]]) vs pairwise indexing ( (x0,x1) = data[index]) on a 2D self.data array (shape=(19115,210))
		#My timings (per epoch) from 100,000 pairs per epoch (note: no difference found using int64 vs int32 vs int16 in torch)
		#Using torch tensor on GPU, indvidual indexing - 6.7 seconds per epoch
		#Using torch tensor on CPU, indvidual indexing - 0.85 seconds per epoch
		#Using torch tensor on GPU, indexing as pair (returns tuple) -- 2 second per epoch
		#Using torch tensor on CPU, indexing as pair (returns tuple) -- 6.3 seconds per epoch 
		#Using numpy, indexing as pair (returns tuple) -- 7.1 seconds per epoch 
		#Using numpy, indvidual indexing -- 0.41 seconds per epoch  (winner)
		#Doing nothing (just indexing the class list and returning) 0.12
		#Thus, numpy is about 3x faster than a cpu tensor, and 6x-20x faster than a gpu tensor (which becomes the bottleneck if left on the gpu)
		#**shrug**
		
		self.pairLst =pairLst#torch.tensor(pairLst).long()
		
		self.noClasses=False
		
		#network runner predict checks for a -1 as the first value to mean no class data provided
		#assigning a list of -1's as class allows us to:
			#avoid an extra if statement in the dataset function
			#ensure that the same amount of data is returned (data,class) in each call even when predicting with no known class
			#ensure the torch collate function will work even when no class data exists
		#please don't use a class value of -1 if intending to pass classes to the predict function (eval)
		if classData is None:
			self.noClasses=True
			self.classData = torch.ones(self.pairLst.shape[0])*-1
		else:
			self.classData = torch.tensor(classData)
		self.classData = self.classData.long()
		
		self.full_gpu = full_gpu #if true, push everything to gpu
		
			
	def __getitem__(self,index):
		y = self.classData[index]
		#individually indexing is faster?
		x0 = self.data[self.pairLst[index][0]]
		x1 = self.data[self.pairLst[index][1]]
		x0 = x0.unsqueeze(0)
		x1 = x1.unsqueeze(0)
		x0 = x0.float()
		x1 = x1.float()
		return (x0,x1,y)
		
	def __len__(self):
		return self.classData.shape[0]


	def activate(self):		
		if self.full_gpu: #push everything to gpu
			self.data = self.data.cuda()
			#self.pairLst = self.pairLst.cuda()
			self.classData = self.classData.cuda()
	
	def deactivate(self):
		self.data = self.data.cpu()
		#self.pairLst = self.pairLst.cpu()
		self.classData = self.classData.cpu()