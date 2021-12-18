#Based on paper Advancing the Prediction Accuracy of Protein-Protein Interactions by Utilizing Evolutionary Information from Position-Specific Scoring Matrix and Ensemble Classifier by Wang, You, Xia, and Liu

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
from sklearn.decomposition import LatentDirichletAllocation


class Wang2017RotFModule(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		self.hyperParams['Model'] = self.hyperParams.get('Model','RotationForest')
		self.hyperParams['n_features_per_subset'] = self.hyperParams.get('n_features_per_subset',80) #5 feature sets, 400 features, so 80 per subset.  Or does paper mean 5 per subset?
		self.hyperParams['max_features'] = self.hyperParams.get('max_featurs',None)
		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',5)
		self.featDict = self.hyperParams.get('featDict',{'all':['PSSMDCT.tsv']})
		
	def genModel(self):
		self.model = GenericForest(self.hyperParams)
