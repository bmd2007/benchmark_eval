import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import time
import torch
import numpy as np
from torch import nn
from torch.utils import data as torchData
import sys
from SimpleDataset import SimpleDataset
import torch.nn.functional as F


class NetworkRunner(object):
	def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=2,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=True,hyp={}):
		self.net = net
		self.hyp = hyp
		
		self.deviceType = hyp.get('deviceType',deviceType)
		
		if deviceType is None or self.deviceType not in ['cpu','cuda']:
			self.deviceType = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.net.to(self.deviceType)
		
		self.predictSoftmax = hyp.get('predictSoftmax',predictSoftmax)
		
		self.batch_size = hyp.get('batchSize',batch_size)
		
		self.num_workers = 0 if sys.platform == 'win32' else 2
		self.epoch = 0
		
		#basic criterion, use set criterion to pass in your own
		self.criterion = torch.nn.CrossEntropyLoss()
		
		optType = hyp.get('optType',optType)
		
		#basic optimizer options, only set for adam and sgd
		#use set optimizer to pass in your own if not using adam or sgd
		#only defining adam and sgd here since they are the most widely used
		#only lr and weight_decay are accessibly from function definition, since they are generic, but more properties can be accessed here
		if optType == 'Adam':
			lr = hyp.get('lr',lr)
			weight_decay = hyp.get('weightDecay',weight_decay)
			eps = hyp.get('eps',1e-8)
			amsgrad = hyp.get('amsgrad',False)
			betas = hyp.get('betas',(0.9,0.9999))
			self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,weight_decay=weight_decay,eps=eps,amsgrad=amsgrad,betas=betas)
			
		elif optType == 'SGD':
			lr = hyp.get('lr',lr)
			weight_decay = hyp.get('weightDecay',weight_decay)
			momentum = hyp.get('momentum',0)
			dampening = hyp.get('dampening',0)
			nesterov = hyp.get('nesterov',False)
			self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum,dampening=dampening,nesterov=nesterov)
		elif optType == 'RMSprop':
			lr = hyp.get('lr',lr)
			weight_decay = hyp.get('weightDecay',weight_decay)
			momentum = hyp.get('momentum',0)
			alpha = hyp.get('alpha',0.99)
			eps = hyp.get('eps',1e-8)
			centered=hyp.get('centered',True)
			self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum,alpha=alpha,eps=eps,centered=centered)
		else:
			self.optimizer = None
			
			
		sched_factor=hyp.get('schedFactor',sched_factor)
		sched_patience=hyp.get('schedPatience',sched_patience)
		sched_cooldown=hyp.get('schedCooldown',sched_cooldown)
		sched_thresh=hyp.get('schedThresh',sched_thresh)
		sched_mode=hyp.get('schedMode','min')
		sched_eps = hyp.get('schedEPS',1e-8)
		sched_verbose = hyp.get('schedVerbose',False)
		sched_thresholdMode = hyp.get('threshSchedMode','rel')
			
		#basic scheduler, using ReduceLROnPlateau, use set schedule to pass in your own
		if sched_factor is None:
			self.scheduler = None
		else:
			self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode=sched_mode,factor = sched_factor,patience = sched_patience, cooldown=sched_cooldown,threshold = sched_thresh, eps=sched_eps,threshold_mode=sched_thresholdMode,verbose=sched_verbose)
		
		
	def getLoaderArgs(self,shuffle=True, pinMem=False):
		return dict(shuffle=shuffle, batch_size=self.batch_size,num_workers = self.num_workers,pin_memory=pinMem)
		
	def setCriterion(self,crit):
		self.criterion = crit
		
	def setOptimizer(self,opt):
		self.optimizer = opt
		
	def setScheduler(self,sch):
		self.scheduler = sch
		
	def getL1LossVal(self):
		l1val = 0
		for name, param in self.net.named_parameters():
			if 'weight' in name or 'bias' in name:
				l1val = l1val + torch.abs(param).sum()
		return l1val

		
	def getLr(self):
		for param_group in self.optimizer.param_groups:
			return param_group['lr']
		
		
	def getLoss(self,output,classData):
		return self.criterion(output,classData)
		
	
	def trainNumpy(self,features,classes,num_iterations,seed=1,min_lr=1e-6,full_gpu=False):
		dataset = SimpleDataset(features,classes,full_gpu)
		self.train(dataset,num_iterations,seed,min_lr)
		
	def updateScheduler(self,lossVal):
		if self.scheduler is not None:
			self.scheduler.step(lossVal)
			
	#min_lr -- minimum learning rate to continue with, only relevant when using scheduler (set sched_factor to none when calling networkrunner init to not use scheduler)
	def train(self,dataset, num_iterations,seed=1,min_lr=1e-6):
		torch.manual_seed(seed)
		self.dataset = dataset

		self.dataset.activate()
		#don't pin memory if pushing entire training set to gpu
		self.curLoader = torchData.DataLoader(self.dataset,**self.getLoaderArgs(True,(not self.dataset.full_gpu)))
		for i in range(0,num_iterations):
			trainLoss = self.train_epoch()
			self.updateScheduler(trainLoss)
			if self.getLr() < min_lr:
				break
				
		self.dataset.deactivate()
		#release memory
		self.dataset = None
		self.curLoader = None
	
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
			classActual = validationDataset.classData.cpu().numpy()
			oneAcc = classActual[classPredictions==1].sum()/max(1,classActual.sum())
			zeroAcc = (1-classActual[classPredictions==0]).sum()/max(1,(1-classActual).sum())
			print('Eval Loss',evalLoss,'class accuracies',zeroAcc,oneAcc)
			self.updateScheduler(evalLoss)
			if self.getLr() < min_lr:
				break
				
		self.dataset.deactivate()
		#release memory
		self.dataset = None
		self.curLoader = None
	
			
	def train_epoch(self):
		self.net.train()
		self.criterion = self.criterion.to(self.deviceType)
		running_loss = 0
		totalPairs = 0
		start_time = time.time()
		for batch_idx, (data, classData) in enumerate(self.curLoader):
			self.optimizer.zero_grad()
			data = data.to(self.deviceType)
			classData = classData.to(self.deviceType)
			out = self.net.forward(data)
			loss = self.getLoss(out,classData)
			running_loss += loss.item()*classData.shape[0]
			totalPairs += classData.shape[0]
			
			loss.backward()
			#torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
			self.optimizer.step()
		end_time = time.time()
		running_loss /= totalPairs
		self.epoch += 1
		
		print('Epoch ',self.epoch, 'Train Loss: ', running_loss, 'LR', self.getLr(),'Time: ',end_time - start_time, 's')
		return running_loss

	
	def processPredictions(self,outputs):
		if self.predictSoftmax:
			outputs = F.softmax(outputs,1)
		return outputs
	
	def predictFromLoader(self,loader):
		outputsLst = []
		runningLoss = 0
		totalPairs = 0
		self.criterion = self.criterion.to(self.deviceType)
		with torch.no_grad():
			self.net.eval()
			for batch_idx, (data, classData) in enumerate(loader):
				data = data.to(self.deviceType)
				outputs = self.net(data)
				if len(classData.shape) > 1 or classData[0]!= -1:
					classData = classData.to(self.deviceType)
					loss = self.getLoss(outputs,classData).detach().item()
				else:
					loss = -1
				runningLoss += loss*data.shape[0]
				totalPairs += data.shape[0]
				
				
				outputs = self.processPredictions(outputs)
	
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
				data = data.to(self.deviceType)
				outputs = self.net(data)
				if len(classData.shape) > 1 or classData[0]!= -1:
					classData = classData.to(self.deviceType)
					loss = self.getLoss(outputs,classData).detach().cpu()#.tolist()
				else:
					loss = torch.ones(data.shape[0])*-1 #just append -1 for each loss
				
				lossVals.append(loss.numpy())
				
				totalPairs += data.shape[0]
				
				
				outputs = self.processPredictions(outputs)
	
				outputs = outputs.to('cpu').detach().numpy()
				outputsLst.append(outputs)
				
			outputsLst = np.vstack(outputsLst)
			lossVals = np.vstack(lossVals)
			self.criterion.reduction=curRed
			return (outputsLst,lossVals)
		
	def predictWithInvLoss(self,predictDataset):
		predictLoader = torchData.DataLoader(predictDataset,**self.getLoaderArgs(False,False))
		return self.predictWithIndvLossFromLoader(predictLoader)
	
	def predict(self,predictDataset):
		predictLoader = torchData.DataLoader(predictDataset,**self.getLoaderArgs(False,False))
		return self.predictFromLoader(predictLoader)
		
	def predictNumpy(self,data, classData=None):
		predictDataset = SimpleDataset(data,classData)
		return self.predict(predictDataset)
		
		
		
	def save(self,fname):
		if self.scheduler:
			state = {'epoch': self.epoch,
			'net': self.net.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'scheduler': self.scheduler.state_dict()
			}
		else:
			state = {'epoch': self.epoch,
			'net': self.net.state_dict(),
			'optimizer': self.optimizer.state_dict()
			}
		torch.save(state,fname)
		
		
	def load(self,fname):
		state = torch.load(fname)
		self.net.load_state_dict(state['net'])
		self.optimizer.load_state_dict(state['optimizer'])
		self.epoch = state['epoch']
		if self.scheduler is not None:
			self.scheduler.load_state_dict(state['scheduler'])
		if self.deviceType == 'cuda':
			for state in self.optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.cuda()


