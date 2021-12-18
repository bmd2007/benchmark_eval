#Based on paper Assessing semantic similarity measures for the characterization of human regulatory pathways by Guo, Liu, Shriver, Hu, and Liebman
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import time
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from GenericPairwisePredictor import GenericPairwisePredictor

class LRModel(object):
	def __init__(self,hyp):
		self.model = LogisticRegression()
	def fit(self,trainFeatures,trainClasses):
		z = self.model.fit(trainFeatures,trainClasses)
		return z
				
	def predict_proba(self,trainFeatures):
		return self.model.predict_proba(trainFeatures)
		
	def loadModel(self,fname):
		self.loadModelFromFile(fname)
	
	def saveModel(self,fname):
		self.saveModelToFile(fname)
		
	def saveModelToFile(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		dump(self.model,fname)

	def loadModelFromFile(self,fname):
		self.model = load(fname)
	

class GouVar2006GOLRModule(GenericPairwisePredictor):
	def __init__(self, hyperParams = None):
		GenericPairwisePredictor.__init__(self,hyperParams)
		
		
		self.featDict=  None
		self.TrainFiles = self.hyperParams.get('TrainFiles',['train_GOSS.tsv'])
		self.TestFiles = self.hyperParams.get('TestFiles',['test_GOSS.tsv'])
		self.ColumnNames = self.hyperParams.get('Columns',['GOSS_BP_Resnik_max','GOSS_CC_Resnik_max','GOSS_MF_Resnik_max'])
		self.AugmentFunction = self.hyperParams.get('Augment',[])
		self.testDataset = self.hyperParams.get('testData','test.tsv')
		self.trainDataset = self.hyperParams.get('trainData','train.tsv')
		self.datasetsHaveHeader = self.hyperParams.get('datasetHeaders',False)
		self.replaceMissing = self.hyperParams.get('replaceMissing',0)
		
		if 'seed' in hyperParams:
			self.seed = int(hyperParams['seed'])
		else:
			self.seed = 1
		
		self.model=None
		self.scaleData = None
		self.featureFolder = None
	
	def genModel(self):
		self.model = LRModel(self.hyperParams)
		
	