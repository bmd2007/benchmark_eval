import numpy as np
import torch
from torch.utils import data

class SimpleDataset(data.Dataset):
	def __init__(self,featureData,classData=None,full_gpu=False):
		self.data = torch.tensor(featureData).float()
		
		#network runner predict checks for a -1 as the first value to mean no class data provided
		#assigning a list of -1's as class allows us to:
			#avoid an extra if statement in the dataset function
			#ensure that the same amount of data is returned (data,class) in each call even when predicting with no known class
			#ensure the torch collate function will work even when no class data exists
		#please don't use a class value of -1 if intending to pass classes to the predict function (eval)
		if classData is not None:
			self.classData = torch.tensor(classData).long()
			self.noClasses=False
		else:
			self.noClasses=True
			self.classData = torch.from_numpy(np.zeros(self.data.shape[0])-1)
		
		self.full_gpu = full_gpu #if true, push everything to gpu
			
	def __getitem__(self,index):
		return (self.data[index,:].flatten().float(),self.classData[index])
		
	def __len__(self):
		return self.data.shape[0]


	def activate(self):
		if self.full_gpu: #push everything to gpu
			self.data = self.data.cuda()
			self.classData = self.classData.cuda()
	
	def deactivate(self):
		self.data = self.data.cpu()
		self.classData = self.classData.cpu()