#Based on paper Predicting Protein-Protein Interactions by Combing Various Sequence- Derived Features into the General Form of Chou’s Pseudo Amino Acid Composition by Zhao, Ma, and Yin

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from GenericSVM import GenericSVM


class Zhao2012SVM(GenericModule):
	def __init__(self, hyperParams = None):
		GenericModule.__init__(self,hyperParams)
		self.PCA = PCA(n_components=67)
		self.scaler = StandardScaler()
		self.modelType = None
		self.featDict = self.hyperParams.get('featDict',{'all':['NMBroto_Zhao_30.tsv', 'Moran_Zhao_30.tsv', 'Geary_Zhao_30.tsv','PSEAAC_Zhao_30.tsv','Grantham_Sequence_Order_30.tsv','Schneider_Sequence_Order_30.tsv','Grantham_Quasi_30.tsv','Schneider_Quasi_30.tsv']})
		
	def genModel(self):
		self.model = GenericSVM(self.hyperParams)

	def loadFeatureData(self,featureFolder):
		super().loadFeatureData(featureFolder)
		self.featuresData['all'].data = self.scaler.fit_transform(self.featuresData['all'].data)
		self.featuresData['all'].data = self.PCA.fit_transform(self.featuresData['all'].data)
		