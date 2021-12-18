#Based on paper Predicting protein–protein interactions by fusing various Chou's pseudo components and using wavelet denoising approach by Tian, Wu, Chen, Qiu, Ma, and Yu

import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import torch
import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
from GenericModule import GenericModule
from joblib import dump, load
from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import StandardScaler
from GenericSVM import GenericSVM

class Tian2019SVM(GenericModule):
	def __init__(self, hyperParams = None):
		if 'featScaleClass' not in hyperParams:
			hyperParams['featScaleClass'] = StandardScaler
		GenericModule.__init__(self,hyperParams)
		self.modelType = None
		
		self.featDict = self.hyperParams.get('featDict',{'all':['EGBW11.tsv','AC11.tsv','PSAAC9.tsv']})
				
	def genModel(self):
		self.model = GenericSVM(self.hyperParams)
	
	#in the paper, denosing was done, presumable on the entire dataset at once.  If we use a large amount of data, we will likely have to do it in batches
	#also, the paper seems to do it on all data (train and test) in one batch, which I'm not doing here, as that can bias the results of the predictions
	def genFeatureData(self,pairs,dataType='Train'):
		classData = pairs[:,2]
		featData = self.featuresData['all'].genData(pairs)
		
		#run data through wavelet denoising
		featData = denoise_wavelet(featData,mode='hard',wavelet='db8',wavelet_levels=6)
		featData = torch.tensor(featData)
		featData = featData - featData.mean(axis=1,keepdims=True)
		featData = featData / torch.sqrt(torch.mul(featData,featData).mean(1)).unsqueeze(0).T
		featData = featData.numpy()
		return featData, classData
	
	