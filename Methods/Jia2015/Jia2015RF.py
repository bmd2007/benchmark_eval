#Based on paper iPPI-Esml: An ensemble classifier for identifying the interactions of proteins by incorporating their physicochemical properties and wavelet transforms into PseAAC by Jia, Liu, Xiao, Liu, and Chou

import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
from GenericModule import GenericModule
from joblib import dump, load
from GenericForest import GenericForest 
import PPIPUtils


class Jia2015Model():
	def __init__(self,hyperParams,numForests=7):
		self.forests = []
		for i in range(0,numForests):
			self.forests.append(GenericForest(hyperParams))
		
	def fit(self,data,classes):
		for i in range(0,len(self.forests)):
			self.forests[i].fit(data[i],classes)
			
	#to convert from voting to probability, we are using the average of all weights as a tie breaker
	def predict_proba(self,data):
		predictions = []
		for i in range(0,len(self.forests)):
			predictions.append(self.forests[i].predict_proba(data[i])[:,1,None])
		predictions = np.hstack(predictions)
		#as the number of forests gets larger, the tiebreaker gets smaller, ensuring tiebreaker never overrides vote
		votes = np.mean((predictions>=0.5),axis=1,keepdims=True) * (1-.1/len(self.forests))
		tiebreaker = np.mean(predictions,axis=1,keepdims=True)*.1/len(self.forests)
		posPreds =  votes + tiebreaker
		negPreds = 1-posPreds
		x = np.hstack((negPreds,posPreds))
		return x
		
	def saveModelToFile(self,fname):
		self.saveAll(fname)
	
	def loadModelFromFile(self,fname):
		self.loadAll(fname)
		
	#save all forests to files
	def saveAll(self,folderName):
		PPIPUtils.makeDir(folderName)
		for i in range(0,len(self.forests)):
			self.forests[i].saveModelToFile(folderName+str(i)+'.out')
			
	#load all forests from files
	def loadAll(self,folderName):
		for i in range(0,len(self.forests)):
			try:
				self.forests[i].loadModelFromFile(folderName+str(i)+'.out')
			except:
				print('Error, cannot find file at location',folderName+str(i)+'.out')
				exit(42)
		

		

class Jia2015RFModule(GenericModule):
	def __init__(self, hyperParams = None, numAttributes=7):
		GenericModule.__init__(self,hyperParams)
		self.numAttributes = self.hyperParams.get('numAttributes',numAttributes)
		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',200)
		self.featDict = self.hyperParams.get('featDict',{'all':['DWTAC.tsv']})
			
		
	def genModel(self):
		self.model = Jia2015Model(self.hyperParams,self.numAttributes)

	def breakIntoGroups(self,features):
		featsPerGroup = features.shape[1]//(self.numAttributes*2)
		feats = []
		for i in range(0,features.shape[1]//2,featsPerGroup):
			feats.append(features[:,i:(i+featsPerGroup)])
		
		idx = 0
		for i in range(features.shape[1]//2,features.shape[1],featsPerGroup):
			feats[idx] = np.hstack((feats[idx],features[:,i:(i+featsPerGroup)]))
			idx+=1
		
		return feats
	
	def fit(self,trainFeatures,trainClasses):
		trainFeatures = self.breakIntoGroups(trainFeatures)
		super().fit(trainFeatures,trainClasses)

	def predict_proba(self,predictFeatures,predictClasses):
		predictFeatures = self.breakIntoGroups(predictFeatures)
		return super().predict_proba(predictFeatures,predictClasses)
