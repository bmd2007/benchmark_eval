
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import time
import numpy as np
from joblib import dump, load
from GenericPairwisePredictor import GenericPairwisePredictor
from GenericForest import GenericForest
class SimpleEnsembleModule(GenericPairwisePredictor):
	def __init__(self, hyperParams = None):
		GenericPairwisePredictor.__init__(self,hyperParams)
		self.featDict=  None
		
		
		self.TrainFiles = self.hyperParams.get('TrainFiles',['train_GOSS.tsv','train_GOL2Pairs.tsv','train_domainPairs.tsv'])
		self.TestFiles = self.hyperParams.get('TestFiles',['test_GOSS.tsv','test_GOL2Pairs.tsv','test_domainPairs.tsv'])
		colLst = ['PFam_All_avgmax','Prosite_All_avgmax','InterPro_All_avgmax','GOL2Freq_All_CC_avgmax','GOL2Freq_All_BP_avgmax','GOL2Freq_All_MF_avgmax','GOL2Freq_All_ALL_avgmax','GOSS_BP_Resnik_avgmax','GOSS_CC_Resnik_avgmax','GOSS_MF_Resnik_avgmax']
		self.ColumnNames = self.hyperParams.get('Columns',colLst)
		self.AugmentFunction = self.hyperParams.get('Augment',[])
		self.testDataset = self.hyperParams.get('testData','test.tsv')
		self.trainDataset = self.hyperParams.get('trainData','train.tsv')
		self.datasetsHaveHeader = self.hyperParams.get('datasetHeaders',False)
		self.replaceMissing = self.hyperParams.get('replaceMissing',0)

		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',30)

		if 'seed' in hyperParams:
			self.seed = int(hyperParams['seed'])
		else:
			self.seed = 1
		
		self.model=None
		self.scaleData = None
		self.featureFolder = None
	
	def genModel(self):
		self.model = GenericForest(self.hyperParams)
		
	

class SimpleEnsembleAllModule(SimpleEnsembleModule):
	def __init__(self,hyperParams=None):
		colLst = ['PFam_All_avgmax','Prosite_All_avgmax','InterPro_All_avgmax','GOL2Freq_All_CC_avgmax','GOL2Freq_All_BP_avgmax','GOL2Freq_All_MF_avgmax','GOL2Freq_All_ALL_avgmax','GOSS_BP_Resnik_avgmax','GOSS_CC_Resnik_avgmax','GOSS_MF_Resnik_avgmax']
		hyperParams['Columns'] = hyperParams.get('Columns',colLst)
		super().__init__(hyperParams)
		
class SimpleEnsembleNonTestModule(SimpleEnsembleModule):
	def __init__(self,hyperParams=None):
		colLst = ['PFam_Non_Test_avgmax','Prosite_All_avgmax','InterPro_Non_Test_avgmax','GOL2Freq_Non_Test_CC_avgmax','GOL2Freq_Non_Test_BP_avgmax','GOL2Freq_Non_Test_MF_avgmax','GOL2Freq_Non_Test_ALL_avgmax','GOSS_BP_Resnik_avgmax','GOSS_CC_Resnik_avgmax','GOSS_MF_Resnik_avgmax']
		hyperParams['Columns'] = hyperParams.get('Columns',colLst)
		super().__init__(hyperParams)
		
class SimpleEnsembleHeldOutModule(SimpleEnsembleModule):
	def __init__(self,hyperParams=None):
		colLst = ['PFam_Heldout_avgmax','Prosite_Heldout_avgmax','InterPro_Heldout_avgmax','GOL2_Freq_Heldout_CC_avgmax','GOL2_Freq_Heldout_BP_avgmax','GOL2_Freq_Heldout_MF_avgmax','GOL2_Freq_Heldout_ALL_avgmax','GOSS_BP_Resnik_avgmax','GOSS_CC_Resnik_avgmax','GOSS_MF_Resnik_avgmax']
		hyperParams['Columns'] = hyperParams.get('Columns',colLst)
		super().__init__(hyperParams)