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

#weight_decay=0.0000005
class NetworkRunnerAuto(NetworkRunner):
	def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=2,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=True,hyp={}):
		NetworkRunner.__init__(self,net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax,hyp)
		self.criterion=torch.nn.MSELoss()
		self.criterion = self.criterion.to(self.deviceType)
	
	def trainNumpy(self,features,num_iterations,seed=1,min_lr=1e-6,full_gpu=False):
		dataset = SimpleAutoDataset(features,full_gpu)
		self.train(dataset,num_iterations,seed,min_lr)
		
	def predictNumpy(self,data):
		predictDataset = SimpleAutoDataset(data)
		return self.predict(predictDataset)
		
	def trainWithValidation(self,dataset,validationDataset,num_iterations,seed=1,min_lr=1e-6):
		torch.manual_seed(seed)
		self.dataset = dataset

		self.dataset.activate()
		validationDataset.activate()
		
		#don't pin memory if pushing entire training set to gpu
		self.curLoader = torchData.DataLoader(self.dataset,**self.getLoaderArgs(True,(not self.dataset.full_gpu)))
		
		#don't pin memory if pushing entire training set to gpu
		validationLoader = torchData.DataLoader(validationDataset,**self.getLoaderArgs(False,(not validationDataset.full_gpu)))
		
		for i in range(0,num_iterations):
			trainLoss = self.train_epoch()
			predictions, evalLoss = self.predictFromLoader(validationLoader)
			classPredictions = predictions.argmax(axis=1)
			#no class data in autoencoder. . .
			#classActual = validationDataset.classData.numpy()
			#oneAcc = classActual[classPredictions==1].sum()/max(1,classActual.sum())
			#zeroAcc = (1-classActual[classPredictions==0]).sum()/max(1,(1-classActual).sum())
			print('Eval Loss',evalLoss)#,'class accuracies',zeroAcc,oneAcc)
			if self.scheduler is not None:
				self.scheduler.step(evalLoss)
			if self.getLr() < min_lr:
				break
				
		self.dataset.deactivate()
		#release memory
		self.dataset = None
		self.curLoader = None
	
			
	