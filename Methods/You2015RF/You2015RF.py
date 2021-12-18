#Based on paper Predicting protein-protein interactions from primary protein sequences using a novel multi-scale local feature representation scheme and the random forest You, Chan, and Hu

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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from GenericSVM import GenericSVM
from GenericForest import GenericForest 
from sklearn.decomposition import LatentDirichletAllocation


class You2015RFModule(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		self.hyperParams['max_features'] = self.hyperParams.get('max_features',10)
		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',60)
		self.featDict = self.hyperParams.get('featDict',{'all':['MLD4_CTD_ConjointTriad_C.tsv','MLD4_CTD_ConjointTriad_T.tsv','MLD4_CTD_ConjointTriad_D.tsv']})
		
	def genModel(self):
		self.model = GenericForest(self.hyperParams)
