import numpy as np
import torch
from torch.utils import data
from joblib import dump, load

#loads from files for each protein instead of memoryview
#full_gpu will try to load all the files into memory, then load all data to gpu on a call to activate
#this probably is a bad idea, since this file is designed for larged datasets that must be stored out of memory
class SledzieskiDataset(data.Dataset):
	def __init__(self, directory, pairLst,classData=None,full_gpu=False):
		self.folderName = directory
		self.data = pairLst
		self.classData = classData
		if self.classData is None:
			self.classData = np.zeros(self.data.shape[0])-1
		self.rawData ={}
		self.full_gpu = full_gpu
		if self.full_gpu:
			for item in self.data:
				for i in range(0,2):
					if item[i] not in rawData:
						self.rawData[item[i]] = self.load(item[i])
	def __getitem__(self,index):
		if self.full_gpu:
			return self.getItemTorch(index)
		else:
			return self.getItemReg(index)
		
	def __len__(self):
		return self.data.shape[0]

	def activate(self):
		if self.full_gpu: #push everything to gpu
			self.torchData ={}
			for item in self.rawData:
				self.torchData[item] = torch.from_numpy(self.data).cuda().float()
			self.torchClasses = torch.from_numpy(self.classData).cuda().long()
	
	def deactivate(self):
		self.torchData = None
		self.torchClasses = None
	
	def getItemTorch(self,index):
		pIdx1, pIdx2 = self.data[index]
		return (self.torchData[pIdx1,:].float(),self.torchData[pIdx2,:].float(),self.torchClasses[index])

	def getItemReg(self,index):
		pIdx1, pIdx2 = self.data[index]
		a = torch.from_numpy(self.load(pIdx1)).float()
		b = torch.from_numpy(self.load(pIdx2)).float()
		return (a,b,torch.from_numpy(np.asarray(self.classData[index])).float().squeeze())


	def load(self,name):
		name = str(name)
		return load(self.folderName+name+'.joblibdump')