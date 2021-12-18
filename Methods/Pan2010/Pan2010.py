#Based on paper Large-scale prediction of human protein-protein interactions from amino acid sequence based on latent topic features by Pan, Zhang, and Shen

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
from sklearn.preprocessing import MinMaxScaler
from GenericSVM import GenericSVM
from GenericForest import GenericForest 
from sklearn.decomposition import LatentDirichletAllocation
from joblib import dump, load


class Pan2010Module(GenericModule):
	def __init__(self, hyperParams = None, scalerType='LDA',featureType='CT',model='RANDOMFOREST'):
		GenericModule.__init__(self,hyperParams)
		
		self.scalerType = self.hyperParams.get('ScalerType',scalerType)
		#lda params
		if self.scalerType == 'LDA':
			topicsVal = self.hyperParams.get('n_components',self.hyperParams.get('LDA_topics',50))
			alphaVal = self.hyperParams.get('doc_topic_prior',self.hyperParams.get('LDA_alpha',50/topicsVal))
			betaVal = self.hyperParams.get('doc_topic_prior',self.hyperParams.get('LDA_beta',0.1))
			self.scaler = LatentDirichletAllocation(n_components=topicsVal,doc_topic_prior=alphaVal,topic_word_prior=betaVal)
		else:
			self.scaler = None

		self.featureType = self.hyperParams.get('FeatureType',featureType).upper()
		if self.featureType not in ['CT','AC','PSAAC']:
			self.featureType = 'CT'
			
		self.modelType = self.hyperParams.get('Model',model).upper()
		if self.modelType in ['LIBSVM','THUNDERSVM','SGDCLASSIFIER','SGD','LINEARSVC','SVC','LIBLINEAR']:
			self.modelType ='SVM'
			if self.featScaleClass is None:
				self.featScaleClass = MinMaxScaler
			
		if self.modelType not in ['ROTATIONFOREST','RANDOMFOREST','SVM']:
			self.modelType='RANDOMFOREST'
			
		
		#svm default parameters
		self.svmParams = self.hyperParams.get('SVMParams',{})
		if '-c' or 'C' in self.svmParams:
			self.svmParams['-c'] = self.svmParams.get('-c',self.svmParams.get('C',8))
		else:
			self.svmParams['-c'] = self.hyperParams.get('-c',self.hyperParams.get('C',8))
		if '-g' or 'gamma' in self.svmParams:
			self.svmParams['-g'] = self.svmParams.get('-g',self.svmParams.get('gamma',2))
		else:
			self.svmParams['-g'] = self.hyperParams.get('-g',self.hyperParams.get('gamma',2))
		
		
		#random forest parameters
		self.randomForestParams = self.hyperParams.get('RandomForestParams',{})
		self.randomForestParams['n_estimators'] = self.randomForestParams.get('n_estimators',self.hyperParams.get('n_estimators',500))
		
		#rotation forest parameters
		self.rotationForestParams = self.hyperParams.get('RotationForestParams',{})
		self.rotationForestParams['n_estimators'] = self.rotationForestParams.get('n_estimators',self.hyperParams.get('n_estimators',80))
		
		
	def genModel(self):
		if self.modelType =='SVM':
			self.model = GenericSVM({**self.hyperParams,**self.svmParams})
		elif self.modelType=='RANDOMFOREST':
			self.model = GenericForest({**self.hyperParams,**self.randomForestParams})
		elif self.modelType=='ROTATIONFOREST':
			self.model = GenericForest({**self.hyperParams,**self.rotationForestParams})
	
	def loadFeatureData(self,featureFolder):
		self.featuresData = {}
		
		if self.featureType =='CT':
			self.featuresData['all'] = ProteinFeaturesHolder([featureFolder + 'conjointTriad.tsv'])
		elif self.featureType == 'AC':
			self.featuresData['all'] = ProteinFeaturesHolder([featureFolder + 'AC30.tsv'])
		elif self.featureType == 'PSAAC':
			self.featuresData['all'] = ProteinFeaturesHolder([featureFolder + 'PSAAC20.tsv'])
		
		if self.scaler is not None:
			self.featuresData['all'].data = self.scaler.fit_transform(self.featuresData['all'].data)
		
	
class Pan2010ModuleLDACTRANDFOREST(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,'LDA','CT','RANDOMFOREST')
		
class Pan2010ModuleLDACTROTFOREST(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,'LDA','CT','ROTATIONFOREST')

class Pan2010ModuleLDACTSVM(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,'LDA','CT','SVC')

class Pan2010ModuleACRANDFOREST(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,None,'AC','RANDOMFOREST')
		
class Pan2010ModuleACROTFOREST(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,None,'AC','ROTATIONFOREST')

class Pan2010ModuleACSVM(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,None,'AC','SVC')

class Pan2010ModulePSAACRANDFOREST(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,None,'PSAAC','RANDOMFOREST')
		
class Pan2010ModulePSAACROTFOREST(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,None,'PSAAC','ROTATIONFOREST')

class Pan2010ModulePSAACSVM(Pan2010Module):
	def __init__(self,hyperParams):
		super().__init__(hyperParams,None,'PSAAC','SVC')
