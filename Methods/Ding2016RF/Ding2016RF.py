#Based on paper Predicting protein-protein interactions via multivariate mutual information of protein sequences by Ding, Tang, and Guo

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


class Ding2016RFModule(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		self.hyperParams['max_features'] = self.hyperParams.get('max_features',25)
		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',500)
		self.featDict = self.hyperParams.get('featDict',{'all':['AAC20.tsv','MMI.tsv','NMBROTO_6_30.tsv']})
		
	def genModel(self):
		self.model = GenericForest(self.hyperParams)
