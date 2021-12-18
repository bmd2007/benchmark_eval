#Based on paper Prediction of Protein-Protein Interactions Using Local Description of Amino Acid Sequence by Zhou, Gao, and Zheng

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
from GenericSVM import GenericSVM


class ZhouSVM(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		self.hyperParams['-c'] = self.hyperParams.get('-c',self.hyperParams.get('C',32))
		self.hyperParams['-g'] = self.hyperParams.get('-g',self.hyperParams.get('gamma',1/32))
		self.modelType = None
		self.featDict = self.hyperParams.get('featDict',{'all':['LD10_CTD_ConjointTriad_C.tsv','LD10_CTD_ConjointTriad_T.tsv','LD10_CTD_ConjointTriad_D.tsv']})
				
	def genModel(self):
		self.model = GenericSVM(self.hyperParams)
