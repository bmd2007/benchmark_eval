#Based on paper Using support vector machine combined with auto covariance to predict protein–protein interactions from protein sequences by Guo, Yu, Wen, and Li

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


class GuoSVM(GenericModule):
	def __init__(self, hyperParams = None):
		if 'featScaleClass' not in hyperParams:
			hyperParams['featScaleClass'] = StandardScaler

		GenericModule.__init__(self,hyperParams)
		self.hyperParams['-c'] = self.hyperParams.get('-c',self.hyperParams.get('C',32))
		self.hyperParams['-g'] = self.hyperParams.get('-g',self.hyperParams.get('gamma',1/32))
		self.featDict = self.hyperParams.get('featDict',{'all':['AC30.tsv']})
		self.modelType = None
		
	def genModel(self):
		self.model = GenericSVM(self.hyperParams)
