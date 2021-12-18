import time
import torch
import numpy as np
from torch import nn
from torch.utils import data as torchData
import sys
from SimpleDataset import SimpleDataset
from SimpleAutoDataset import SimpleAutoDataset
import torch.nn.functional as F
from NetworkRunner import NetworkRunner

#Network runner that Collates X values without touching them, for data that is variable length
class NetworkRunnerCollate(NetworkRunner):
	def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=1,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=True,hyp={}):
		NetworkRunner.__init__(self,net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax,hyp)
		
	def getLoaderArgs(self,shuffle=True, pinMem=False):
		d = super().getLoaderArgs(shuffle,pinMem)
		d['collate_fn'] = self.collate
		return d
	
	#same as regular train, but doesn't move data to device	
	def train_epoch(self):
		self.net.train()
		self.criterion = self.criterion.to(self.deviceType)
		running_loss = 0
		start_time = time.time()
		totalPairs = 0
		for batch_idx, (data, classData) in enumerate(self.curLoader):
			self.optimizer.zero_grad()
			#data = data.to(self.deviceType)
			classData = classData.to(self.deviceType)
			out = self.net.forward(data)
			loss = self.getLoss(out,classData)
			running_loss += loss.item()*classData.shape[0]
			totalPairs += classData.shape[0]
			
			# backprop loss
			loss.backward()
			
			self.optimizer.step()
			
		end_time = time.time()
		running_loss /= totalPairs
		self.epoch += 1
		print('Epoch ',self.epoch, 'Train Loss: ', running_loss, 'LR', self.getLr(),'Time: ',end_time - start_time, 's')
		return running_loss
		
	#same as regular predict, but doesn't move data to device	
	def predictFromLoader(self,loader):
		outputsLst = []
		runningLoss = 0
		totalPairs = 0
		self.criterion = self.criterion.to(self.deviceType)
		with torch.no_grad():
			self.net.eval()
			for batch_idx, (data, classData) in enumerate(loader):
	#			data = data.to(self.deviceType)
				outputs = self.net(data)
				if classData[0]!= -1:
					classData = classData.to(self.deviceType)
					loss = self.getLoss(outputs,classData).detach().item()
				else:
					loss = -1
				
				outputs = self.processPredictions(outputs)
				
				runningLoss += loss*outputs.shape[0]
				totalPairs += outputs.shape[0]
				
				outputs = outputs.to('cpu').detach().numpy()
				outputsLst.append(outputs)
				
				
			runningLoss /= totalPairs
			outputsLst = np.vstack(outputsLst)
			return (outputsLst,runningLoss)
			
	def predictWithIndvLossFromLoader(self,loader):
		outputsLst = []
		lossVals = []
		totalPairs = 0
		curRed = self.criterion.reduction
		#switch criterion to not reduce, to get per element losses
		self.criterion.reduction='none'
		self.criterion = self.criterion.to(self.deviceType)
		with torch.no_grad():
			self.net.eval()
			for batch_idx, (data, classData) in enumerate(loader):
				#data = data.to(self.deviceType)
				outputs = self.net(data)
				if classData[0]!= -1:
					classData = classData.to(self.deviceType)
					loss = self.getLoss(outputs,classData).detach().tolist()
				else:
					loss = [-1]*data.shape[0] #just append -1 for each loss
				lossVals.extend(loss)
				totalPairs += data.shape[0]
				
				
				outputs = self.processPredictions(outputs)
	
				outputs = outputs.to('cpu').detach().numpy()
				outputsLst.append(outputs)
				
			outputsLst = np.vstack(outputsLst)
			self.criterion.reduction=curRed
			return (outputsLst,lossVals)
		
	def predictWithInvLoss(self,predictDataset):
		predictLoader = torchData.DataLoader(predictDataset,**self.getLoaderArgs(False,False))
		return self.predictWithIndvLossFromLoader
	

	def collate(self,tuples):
		lst = []
		lst[:] = zip(*tuples)
		classes = lst[-1]
		lst = lst[:-1]
		for i in range(0,len(lst)):
			lst[i] = torch.vstack(lst[i]).to(self.deviceType)
		classes = torch.vstack(classes).squeeze(1)
		return (lst,classes)
		
