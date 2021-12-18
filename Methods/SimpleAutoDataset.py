import numpy as np
import torch
from torch.utils import data


class SimpleAutoDataset(data.Dataset):
	def __init__(self,numpyFeatures,full_gpu=False):
		self.data = numpyFeatures
		self.full_gpu = full_gpu #if true, push everything to gpu
			
	def __getitem__(self,index):
		if self.full_gpu:
			return self.getItemTorch(index)
		else:
			return self.getItemReg(index)
		
	def __len__(self):
		return self.data.shape[0]

	def activate(self):
		if self.full_gpu: #push everything to gpu
			self.torchData = torch.from_numpy(self.data).cuda().float()
	
	def deactivate(self):
		self.torchData = None
	
	def getItemTorch(self,index):
		return (self.torchData[index,:].flatten().float(),self.torchData[index,:].flatten().float())

	def getItemReg(self,index):
		return (torch.from_numpy(self.data[index,:]).flatten().float(),torch.from_numpy(self.data[index,:]).flatten().float())
