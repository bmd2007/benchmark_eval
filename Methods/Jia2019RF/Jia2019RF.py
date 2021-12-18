#Based on paper iPPI-PseAAC(CGR): Identify protein-protein interactions by incorporating chaos game representation into PseAAC by Jia, Li, Qiu, Xiao, and Chou

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
from GenericModule import GenericModule
from GenericForest import GenericForest 


class Jia2019RFModule(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',200)
		self.featDict = self.hyperParams.get('featDict',{'all':['AAC20.tsv','Chaos.tsv']})
		
	def genModel(self):
		self.model = GenericForest(self.hyperParams)
