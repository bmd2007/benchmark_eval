#Based on paper Prediction of protein–protein interactions using random decision forest framework by Chen and Liu

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
from GenericForest import GenericForest
import PPIPUtils
from scipy.sparse import csr_matrix

class Chen2005RFModule(GenericModule):
	def __init__(self, hyperParams = None,prioritizeDepth=True):

		GenericModule.__init__(self,hyperParams)
		self.hyperParams['min_samples_split'] = self.hyperParams.get('min_samples_split',3)
		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',150)
		self.hyperParams['max_depth'] = self.hyperParams.get('max_depth',450)
		#self.hyperParams['min_impurity_decrease'] = self.hyperParams.get('min_impurity_decrease',0.01)
		d = {}
		d['DomainLst'] = ['DomainAggs/PfamProteins.tsv']
		d['ProteinMapping'] = [None]
		
		if 'featDict' in self.hyperParams:
			for item in self.hyperParams.featDict:
				d[item] = self.hyperParams.featDict[item]
		self.featDict = d
		self.modelType = None
		
		
		
	def loadFeatureData(self,featureFolder):
		if featureFolder[-1] not in ['/','\\']:
			featureFolder+='/'
		
		domainData = PPIPUtils.parseTSV(featureFolder+self.featDict['DomainLst'][0])
		
		domainIdxMap = {}
		proteinToDomain = {}
		#parse protein to domain mapping
		for line in domainData[1:]:
			protein = line[0]
			domain = line[1]
			if domain not in domainIdxMap:
				domainIdxMap[domain] = len(domainIdxMap)
			domainIdx = domainIdxMap[domain]
			if protein not in proteinToDomain:
				proteinToDomain[protein] = set()
			proteinToDomain[protein].add(domainIdx)
		
		#parse mapping (such as uniprot to entrez) if necessary
		if self.featDict['ProteinMapping'][0] is not None:
			protMappingData = PPIPUtils.parseTSV(featureFolder+self.featDict['ProteinMapping'][0])
			domainMapping = {}
			for line in protMappingData[1:]:
				prot = line[0]
				mapping = line[1]
				if prot not in domainMapping:
					domainMapping[prot] = set()
				domainMapping[prot] |= proteinToDomain.get(mapping,set())
		else:
			domainMapping = proteinToDomain
		
		
		#create matrix of protein domain matches, and protMap mapping proteins to rows
		#leave row 0 blank (all zeros), and map all values not in protMap to row 0
		self.domainMat = np.zeros((len(domainMapping)+1,len(domainIdxMap)))
		self.protMap = {}
		for item, domIdxs in domainMapping.items():
			protIdx = len(self.protMap)+1
			self.protMap[str(item)] = protIdx
			self.domainMat[protIdx,list(domIdxs)] = 1
		


	def calcDomainPair(self,p1,p2):
		lst = []
		p1 = str(p1)
		p2 = str(p2)
		
		p1Idx = self.protMap.get(p1,-1)
		p2Idx = self.protMap.get(p2,-1)
		
		pLst = []
		for item in [p1Idx,p2Idx]:
			if item == -1:
				pLst.append(np.zeros(self.domainMat.shape[1]))
			else:
				pLst.append(self.domainMat[item,:])
		pLst = np.vstack(pLst).sum(axis=0)
		return pLst
		

	#swap out pair data for their indices in the domain matrix, add together the domain matrix rows, and return the features
	def genFeatureData(self,pairs,dataType='train'):	
		classData = np.asarray(pairs[:,2],dtype=np.int32)
		orgFeatsData = pairs[:,0:2]
		#replace all pair values with their matrix index value
		featsData = [self.protMap.get(str(a),0) for a in orgFeatsData.flatten()]
		featsData = np.asarray(featsData).reshape(classData.shape[0],2)
		
		
		#calculate sum of domain values from domain matrix for all mapping pairs
		values = self.domainMat[featsData[:,0],:] + self.domainMat[featsData[:,1],:]
		return values, classData
			
	def predictPairs(self, testPairs):
		if len(testPairs) > 100:
			return self.predictFromBatch(testPairs,64)
		else:
			return super().predictPairs(testPairs)

		
	def scaleFeatures(self,features,scaleType):
		return features #no scaling
		
	def setScaleFeatures(self,trainPairs):
		pass
	

	def saveFeatScaler(self,fname):
		pass
			
	def loadFeatScaler(self,fname):
		pass
		
	def genModel(self):
		self.model = GenericForest(self.hyperParams)
