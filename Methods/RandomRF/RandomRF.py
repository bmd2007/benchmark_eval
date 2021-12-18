#Based on paper Using support vector machine combined with auto covariance to predict protein–protein interactions from protein sequences by Yanzhi Guo, Lezheng Yu, Zhining Wen, and Menglong Li

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


class RandomRFModule(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		self.featDict = self.hyperParams.get('featDict',{'all':['Random500.tsv']})
		
	def genModel(self):
		self.model = GenericForest(self.hyperParams)
