#Based on paper Protein-protein interaction inference based on semantic similarity of Gene Ontology terms by Zhang and Tang
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
from GenericSVM import GenericSVM
class Zhang2016GOModule(GenericPairwisePredictor):
	def __init__(self, hyperParams = None):
		GenericPairwisePredictor.__init__(self,hyperParams)
		self.featDict=  None
		self.TrainFiles = self.hyperParams.get('TrainFiles',['train_GOSS.tsv'])
		self.TestFiles = self.hyperParams.get('TestFiles',['test_GOSS.tsv'])
		lst = []
		for item in ['GOSS_','GOSS_Desc_']:
			for item2 in ['BP_','CC_','MF_']:
				for item3 in ['Resnik_','Lin_','Jiang_','Rev_','Wu_']:
					lst.append(item+item2+item3+'avgmax')
		self.ColumnNames = self.hyperParams.get('Columns',lst)
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
		self.model = GenericSVM(self.hyperParams)
		
	