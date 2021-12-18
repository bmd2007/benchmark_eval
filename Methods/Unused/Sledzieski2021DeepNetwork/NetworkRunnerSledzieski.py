#Code based on python DScript library (pip install dscript )
#This file utilizes the loss calculations from the train.py file of dscript, as well as the minibatch code
#MIT License

#Copyright (c) 2020 Samuel Sledzieski

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.




import time
import torch
import numpy as np
from torch import nn
from torch.utils import data as torchData
import sys
from SimpleDataset import SimpleDataset
from SimpleAutoDataset import SimpleAutoDataset
import torch.nn.functional as F
from NetworkRunnerCollate import NetworkRunnerCollate

#currently, D-Script code is designed to run 1 pair at a time (not a batch)
#This is probably due to the variable length of the proteins, and the convolution matrix created with 2 variable length dimensions
#Will keep an eye on https://github.com/pytorch/nestedtensor, but I'm not sure if it will work with convolutions once completed or not
#if it works, will have to write new dataset, as one I'm using here expects fixed sizes
#for now, only 1 pair are run through the forward pass at a time (in the network code)
class NetworkRunnerSledzieski(NetworkRunnerCollate):
	def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=1,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=False,hyp={},simObjective=0.35):
		NetworkRunnerCollate.__init__(self,net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax,hyp)
		
		self.criterion = torch.nn.BCELoss();
		self.criterion = self.criterion.to(self.deviceType)
		self.simObjective = hyp.get('simObjective',simObjective)
	
	def getLoss(self,output,classData):
		(c_map_mag, p_hat) = output
		#get bce_loss
		bce_loss = self.criterion(p_hat,classData)
		#get cmap_loss
		cmap_loss = torch.mean(c_map_mag)
		loss = (self.simObjective * bce_loss) + ((1 - self.simObjective) * cmap_loss)
		return loss

	def processPredictions(self,outputs):
		outputs = outputs[1] #only p_hat is returned, not c_map_mag
				
		#get class 0 and class 1 probabilities, to match with other classifiers
		#assuming outputs are in range 0-1, which appears to be the case as the logistic activation is clamped
		#uncertain these are actual probabiltiies
		outputs =torch.cat((1-outputs,outputs),dim=1)
			
		#don't use softmax, outputs are already scaled to 1
		#if self.predictSoftmax:
		#	outputs = F.softmax(outputs,1)
		return outputs
				

	def predictWithIndvLossFromLoader(self,loader):
		pass #undefined
		
	def predictWithInvLoss(self,predictDataset):
		pass #undefined