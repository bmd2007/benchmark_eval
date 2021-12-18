#Based loosely on/inspire by previous work  Prediction of human protein-protein interaction by a domain-based approach by Zhang, Jiao, Song, Chang
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


class ZhangDomainVar2016Module(GenericPairwisePredictor):
	def __init__(self, hyperParams = None):
		GenericPairwisePredictor.__init__(self,hyperParams)
		self.featDict=  None
		self.TrainFiles = self.hyperParams.get('TrainFiles',['train_domainPairs.tsv'])
		self.TestFiles = self.hyperParams.get('TestFiles',['test_domainPairs.tsv'])
		self.ColumnNames = self.hyperParams.get('Columns',['PFam_Non_Test_prod','Prosite_Non_Test_prod','InterPro_Non_Test_prod'])
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
		
	

class ZhangDomainVar2016AllModule(ZhangDomainVar2016Module):
	def __init__(self,hyperParams=None):
		hyperParams['Columns'] = hyperParams.get('Columns',['PFam_All_prod','Prosite_All_prod','InterPro_All_prod'])
		super().__init__(hyperParams)
		
class ZhangDomainVar2016NonTestModule(ZhangDomainVar2016Module):
	def __init__(self,hyperParams=None):
		hyperParams['Columns'] = hyperParams.get('Columns',['PFam_Non_Test_prod','Prosite_Non_Test_prod','InterPro_Non_Test_prod'])
		super().__init__(hyperParams)
		
class ZhangDomainVar2016HeldOutModule(ZhangDomainVar2016Module):
	def __init__(self,hyperParams=None):
		hyperParams['Columns'] = hyperParams.get('Columns',['PFam_Heldout_prod','Prosite_Heldout_prod','InterPro_Heldout_prod'])
		super().__init__(hyperParams)