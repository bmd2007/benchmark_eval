#Based on paper AutoPPI: An Ensemble of Deep Autoencoders for Protein-Protein Interaction Prediction by Czibula, Albu, Bocicor, and Chira

import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import PPIPUtils
import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
import torch
from NetworkRunnerAuto import NetworkRunnerAuto
from SimpleAutoDataset import SimpleAutoDataset
from GenericModule import GenericModule
from GenericNetworkModel import GenericNetworkModel
import torch
import torch.nn as nn
import torch.nn.functional as F

#note, may convert to using a dictionary dataset later for more efficient memory usage

class CzibulaAutoPPIJointJoint(nn.Module):
	def __init__(self,featureSize=763,regSize=600,minSize=300,seed=1):
		super(CzibulaAutoPPIJointJoint, self).__init__()
		torch.manual_seed(seed)
		
		self.featureSize = featureSize
		#encode
		self.EncL1 = nn.Linear(featureSize*2,regSize)
		self.EncL2 = nn.Linear(regSize,regSize)
		self.EncL3 = nn.Linear(regSize,minSize)
		
		#decode
		self.DecL1 = nn.Linear(minSize,regSize)
		self.DecL2 = nn.Linear(regSize,regSize)
		self.DecL3 = nn.Linear(regSize,featureSize*2)
		
		self.act = nn.SELU()
		
	def forward(self,x):
		x = self.EncL1(x)
		x = self.act(x)
		x = self.EncL2(x)
		x = self.act(x)
		x = self.EncL3(x)
		x = self.act(x)
		
		x = self.DecL1(x)
		x = self.act(x)
		x = self.DecL2(x)
		x = self.act(x)
		x = self.DecL3(x)
		return x


class CzibulaAutoPPISiameseJoint(nn.Module):
	def __init__(self,featureSize=763,regSize=600,minSize=300,seed=1):
		super(CzibulaAutoPPISiameseJoint, self).__init__()
		torch.manual_seed(seed)
		
		self.featureSize = featureSize
		#encode
		self.EncL1 = nn.Linear(featureSize,regSize)
		self.EncL2 = nn.Linear(regSize,regSize)
		self.EncL3 = nn.Linear(regSize,minSize)
		
		#decode
		self.DecL1 = nn.Linear(minSize*2,regSize)
		self.DecL2 = nn.Linear(regSize,regSize)
		self.DecL3 = nn.Linear(regSize,featureSize*2)
		
		self.act = nn.SELU()
		
	def forward(self,x):
		x1 = x[:,:self.featureSize]
		x2 = x[:,self.featureSize:]
		dataLst = []
		for item in [x1,x2]:
			item = self.EncL1(item)
			item = self.act(item)
			item = self.EncL2(item)
			item = self.act(item)
			item = self.EncL3(item)
			item = self.act(item)
			dataLst.append(item)
		(x1,x2) = dataLst
		
		x = torch.cat((x1,x2),dim=1)
		
		x = self.DecL1(x)
		x = self.act(x)
		x = self.DecL2(x)
		x = self.act(x)
		x = self.DecL3(x)
		return x
		
	

class CzibulaAutoPPISiameseSiamese(nn.Module):
	def __init__(self,featureSize=763,regSize=600,minSize=300,seed=1):
		super(CzibulaAutoPPISiameseSiamese, self).__init__()
		torch.manual_seed(seed)
		
		self.featureSize = featureSize
		#encode
		self.EncL1 = nn.Linear(featureSize,regSize)
		self.EncL2 = nn.Linear(regSize,regSize)
		self.EncL3 = nn.Linear(regSize,minSize)
		
		#decode
		self.DecL1 = nn.Linear(minSize*2,regSize)
		self.DecL2 = nn.Linear(regSize,regSize)
		self.DecL3 = nn.Linear(regSize,featureSize)
		
		self.act = nn.SELU()
		
	def forward(self,x):
		x1 = x[:,:self.featureSize]
		x2 = x[:,self.featureSize:]
		dataLst = []
		for item in [x1,x2]:
			item = self.EncL1(item)
			item = self.act(item)
			item = self.EncL2(item)
			item = self.act(item)
			item = self.EncL3(item)
			item = self.act(item)
			dataLst.append(item)
		(x1,x2) = dataLst
		
		xNew = torch.mul(x1,x2)
		
		dataLst = []
		x1 = torch.cat((x1,xNew),dim=1)
		x2 = torch.cat((x2,xNew),dim=1)
		for item in [x1,x2]:
			item = self.DecL1(item)
			item = self.act(item)
			item = self.DecL2(item)
			item = self.act(item)
			item = self.DecL3(item)
			dataLst.append(item)
		(x1,x2) = dataLst
		
		out = torch.cat((x1,x2),dim=1)
		return out 
		
class CzibulaNetworkRunner(NetworkRunnerAuto):
	def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=2,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=True,hyp={}):
		NetworkRunnerAuto.__init__(self,net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax,hyp)

	#save memory by only keeping what we need, the average loss value per output
	def predictWithIndvLossFromLoader(self,loader):
		lossVals = []
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
				
				loss = loss.mean(dim=1,keepdims=True)
				lossVals.append(loss.numpy())
			outputsLst = []
			lossVals = np.vstack(lossVals)
			self.criterion.reduction=curRed
			return (outputsLst,lossVals)


	
	

#hyperparams:
class Czibula2021AutoPPIModel(GenericNetworkModel):
	def __init__(self,hyp={},modelType='JJ',featureSize=763,fullGPU=False,deviceType=None,numEpochs=2000,batchSize=64,lr=5e-4,schedFactor=0.5,schedThresh=1e-2,schedPatience=2,minLr=1e-5):
		GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,schedFactor=schedFactor,schedThresh=schedThresh,schedPatience=schedPatience,minLr=minLr)
	
		self.modelType = hyp.get('modelType',modelType).upper()
		if self.modelType not in ['JJ','SS','SJ']:
			self.modelType = 'JJ'
		self.featureSize = hyp.get('featureSize',featureSize)
		self.regSize = hyp.get('regSize',600)
		self.minSize = hyp.get('minSize',300)
		self.model = None
	
	def saveModelToFile(self,fname):
		if self.modelPos is None:
			print('Error, no model to save')
			exit(42)
		self.modelPos.save(fname+'_pos')
		self.modelNeg.save(fname+'_neg')
		
	def genModel(self):
		if self.modelType == 'JJ':
			self.netPos = CzibulaAutoPPIJointJoint(self.featureSize,self.regSize,self.minSize,seed=self.seed)
			self.netNeg = CzibulaAutoPPIJointJoint(self.featureSize,self.regSize,self.minSize,seed=self.seed)
		elif self.modelType == 'SJ':
			self.netPos = CzibulaAutoPPISiameseJoint(self.featureSize,self.regSize,self.minSize,seed=self.seed)
			self.netNeg = CzibulaAutoPPISiameseJoint(self.featureSize,self.regSize,self.minSize,seed=self.seed)
		elif self.modelType == 'SS':
			self.netPos = CzibulaAutoPPISiameseSiamese(self.featureSize,self.regSize,self.minSize,seed=self.seed)
			self.netNeg = CzibulaAutoPPISiameseSiamese(self.featureSize,self.regSize,self.minSize,seed=self.seed)
			
		self.modelPos = CzibulaNetworkRunner(self.netPos,hyp=self.hyp)
		self.modelNeg = CzibulaNetworkRunner(self.netNeg,hyp=self.hyp)
		
	def loadModelFromFile(self,fname):
		if self.model is None:
			self.genModel()
		self.modelPos.load(fname+'_pos')
		self.modelNeg.load(fname+'_neg')

	#train network
	def fit(self,trainFeatures,classes,validationFeatures=None, validationClasses=None):
		self.genModel()
		dataSetTrainPos = SimpleAutoDataset(trainFeatures[classes==1],full_gpu=self.fullGPU)
		dataSetTrainNeg = SimpleAutoDataset(trainFeatures[classes==0],full_gpu=self.fullGPU)
		if validationFeatures is None:
			self.modelPos.train(dataSetTrainPos, self.numEpochs,seed=self.seed,min_lr=self.minLr)
			self.modelNeg.train(dataSetTrainNeg, self.numEpochs,seed=self.seed,min_lr=self.minLr)
		else:
			dataSetValPos = SimpleAutoDataset(validationFeatures[validationClasses==1],full_gpu=self.fullGPU)
			dataSetValNeg = SimpleAutoDataset(validationFeatures[validationClasses==0],full_gpu=self.fullGPU)
			self.modelPos.trainWithValidation(dataSetTrainPos,dataSetValPos,self.numEpochs,seed=self.seed,min_lr=self.minLr)
			self.modelNeg.trainWithValidation(dataSetTrainNeg,dataSetValNeg,self.numEpochs,seed=self.seed,min_lr=self.minLr)
		
		
	#predict on network
	def predict_proba(self,predictDataset):
		dataSetPred = SimpleAutoDataset(predictDataset)
		output,lossPos = self.modelPos.predictWithInvLoss(dataSetPred)
		output,lossNeg = self.modelNeg.predictWithInvLoss(dataSetPred)
		
		#get the loss from the positive and negative autoencoder by averaging the loss across each data pair
		lossPos = torch.tensor(lossPos)
		lossNeg = torch.tensor(lossNeg)
		#concatenate the two togeter, and use softmax for probability
		#paper only states to choose the smaller loss, but wheen need probabilities for our computations
		
		#ordering for probabilities is (class0,class1) or (negative, positive), but, since smaller loss is better, and we prioritizes a higher score by default
		#we are inverting it here, counting lossPos as negative, lossNeg as positive, and choosing the larger score for classification
		loss = torch.cat((lossPos,lossNeg),dim=1)
		probs = F.softmax(loss,1)
		prot = probs.cpu().numpy()
		return probs
		
		

class Czibula2021AutoPPIModule(GenericModule):
	def __init__(self, hyperParams = None,modelType='JJ',featureSize=763,validationRate=0.1):
		GenericModule.__init__(self,hyperParams)
		self.modelType = self.hyperParams.get('modelType',modelType)
		self.featureSize = self.hyperParams.get('featureSize',featureSize)
		self.validationRate = self.hyperParams.get('ValidationRate',validationRate)
		self.featDict = self.hyperParams.get('featDict',{'all':['AC14_30.tsv','ConjointTriad.tsv']})
		
	def genModel(self):
		self.model = Czibula2021AutoPPIModel(self.hyperParams,self.modelType,self.featureSize)

		
	#when running fit or predict, pass in the dataMatrix
	def fit(self,trainFeatures,trainClasses):
		if self.validationRate is not None and self.validationRate > 0:
			newTrainFeat, newTrainClass, newValidFeat, newValidClass = self.splitFeatures(trainFeatures,trainClasses,self.validationRate)
			self.genModel() #create a new model from scratch, ensuring we don't overwrite the previously trained one
			self.model.fit(newTrainFeat,newTrainClass, newValidFeat, newValidClass)
		else:
			self.genModel() #create a new model from scratch, ensuring we don't overwrite the previously trained one
			self.model.fit(trainFeatures,trainClasses)
	
class Czibula2021AutoPPIModuleJJ(Czibula2021AutoPPIModule):
	def __init__(self, hyperParams = None,featureSize=763,validationRate=0.1):
		super().__init__(hyperParams,'JJ',featureSize,validationRate)

class Czibula2021AutoPPIModuleSJ(Czibula2021AutoPPIModule):
	def __init__(self, hyperParams = None,featureSize=763,validationRate=0.1):
		super().__init__(hyperParams,'SJ',featureSize,validationRate)

class Czibula2021AutoPPIModuleSS(Czibula2021AutoPPIModule):
	def __init__(self, hyperParams = None,featureSize=763,validationRate=0.1):
		super().__init__(hyperParams,'SS',featureSize,validationRate)
