#Based on paper Prediction of Protein-Protein Interactions Using An Effective Sequence Based Combined Method by Goktepe and Kodaz

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
from sklearn.preprocessing import MinMaxScaler
from GenericSVM import GenericSVM
from sklearn.decomposition import PCA



class Goktepe2018SVM(GenericModule):
	def __init__(self, hyperParams = None):
#		if 'featScaleClass' not in hyperParams:
#			hyperParams['featScaleClass'] = MinMaxScaler
		self.scaler = hyperParams.get('pcaScaler',MinMaxScaler)
		GenericModule.__init__(self,hyperParams)
		
		self.hyperParams['-c'] = self.hyperParams.get('-c',self.hyperParams.get('C',32))
		self.hyperParams['-g'] = self.hyperParams.get('-g',self.hyperParams.get('gamma',0.04))
		
		componentsDPC = self.hyperParams.get('compDPC',20)
		componentsSkipWeighted = self.hyperParams.get('compSkipWeighted',20)
		componentsPSAAC = self.hyperParams.get('compPSAAC',20)
		componentsFinal = self.hyperParams.get('componentsFinal',390)
		
		self.PCADPC = PCA(n_components=componentsDPC)
		self.PCASkipWeighted = PCA(n_components=componentsSkipWeighted)
		self.PCAPSAAC = PCA(n_components=componentsPSAAC)
		self.PCAFinal = PCA(n_components=componentsFinal)
		self.modelType = None
		
	def genModel(self):
		self.model = GenericSVM(self.hyperParams)
		
	def loadFeatureData(self,featureFolder):
		self.featuresData = {}
		#get data for 3 different features
		psaac20 = ProteinFeaturesHolder([featureFolder + 'PSAAC20.tsv'])
		skipWeighted = ProteinFeaturesHolder([featureFolder + 'SkipWeightedConjointTriad.tsv'])
		pssmDPC = ProteinFeaturesHolder([featureFolder + 'PSSMDPC.tsv'])
		scaler = self.scaler()
		
		psaac20.data = scaler.fit_transform(psaac20.data)
		skipWeighted.data = scaler.fit_transform(skipWeighted.data)
		pssmDPC.data = scaler.fit_transform(pssmDPC.data)
		
		#apply 3 rounds of PCA, 1 per feature
		pca1 = self.PCAPSAAC.fit_transform(psaac20.data)
		pca2 = self.PCASkipWeighted.fit_transform(skipWeighted.data)
		pca3 = self.PCADPC.fit_transform(pssmDPC.data)
		
		#concat all feature with their PCA vectors
		fullData = np.hstack((psaac20.data,skipWeighted.data,pssmDPC.data,pca1,pca2,pca3))
		fullData = scaler.fit_transform(fullData)
		
		#run final round of PCA
		fullData = self.PCAFinal.fit_transform(fullData)
		
		#save full dataset
		psaac20.data =fullData
		self.featuresData['all'] = psaac20
		
