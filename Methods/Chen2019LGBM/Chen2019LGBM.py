#Based on paper LightGBM-PPI: Predicting protein-protein interactions through LightGBM with multi-information fusion by Chen, Zhang, Ma, and Yu


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
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV



class ElasticSelect(object):
	#need to turn hyperParams down to low numbers to allow for the detection of any trends
	#Features should be scaled prior to using elasticnet to ensure they are within a similar range, but I don't see any scaling in the paper
	#All our features or somewhat low numbers (usually positive/negative single digits), so it should be too bad
	def __init__(self,alpha=.01,l1_ratio=0.1,normalize=False):
		self.net = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,normalize=normalize) 
		self.mask=None

	def fit(self,X,y):
		self.net.fit(X,y)
		self.filterLst = self.net.coef_ != 0
		
	def fit_transform(self,X,y):
		self.fit(X,y)
		return self.transform(X)
		
	def transform(self,X):
		return X[:,self.filterLst]

class Chen2019LGBMModule(GenericModule):
	def __init__(self, hyperParams = None):
		hyperParams['featScaleClass'] = hyperParams.get('featScaleClass',ElasticSelect)
		GenericModule.__init__(self,hyperParams)
		self.hyperParams['Model'] = self.hyperParams.get('Model','LGBM')
		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',500)
		self.hyperParams['max_depth'] = self.hyperParams.get('max_depth',15)
		self.hyperParams['learning_rate'] = self.hyperParams.get('learning_rate',0.2)
		self.featDict = self.hyperParams.get('featDict',{'all':['NMBROTO_9.tsv','MORAN_9.tsv','GEARY_9.tsv','PSEAAC_3.tsv','LD10_CTD_ConjointTriad_C.tsv','LD10_CTD_ConjointTriad_T.tsv','LD10_CTD_ConjointTriad_D.tsv','conjointTriad.tsv']})
		
	def genModel(self):
		self.model = GenericForest(self.hyperParams)
		
		
	def train(self,trainPairs):
		trainFeatures, trainClasses = self.genFeatureData(trainPairs,'train')
		trainFeatures = self.scaleFeatures(trainFeatures,'train',trainClasses)
		self.fit(trainFeatures,trainClasses)
		
		
	def scaleFeatures(self,features,scaleType,featClasses=None):
		if self.featScaleClass is not None:
			#iterate through feature dictionary, transforming each group of features given the scaler
			if scaleType == 'train':
				self.scaleModels = self.featScaleClass()
				newFeatures = self.scaleModels.fit_transform(features,featClasses)
				return newFeatures
				
			else:
				newFeatures = self.scaleModels.transform(features)
				return newFeatures
				
		else:#no scaler, do nothing
			return features

	def setScaleFeatures(self,trainPairs):
		pass

