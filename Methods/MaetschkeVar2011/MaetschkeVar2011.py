#Based on paper Gene Ontology-driven inference of protein-protein interactions using inducers by Maetschke, Simonsen, Davis, and Ragan

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
import PPIPUtils
from scipy.sparse import csr_matrix

class MaetschkeVar2011Module(GenericModule):
	def __init__(self, hyperParams = None,prioritizeDepth=True):

		GenericModule.__init__(self,hyperParams)
		self.hyperParams['max_features'] = self.hyperParams.get('max_features',200)
		self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',200)
		#self.hyperParams['n_estimators'] = self.hyperParams.get('n_estimators',100)
		self.hyperParams['max_depth'] = self.hyperParams.get('max_depth',None)
		#self.hyperParams['max_depth'] = self.hyperParams.get('max_depth',20)
		d = {}
		d['GOAncestorsMF'] = ['GeneOntologyAggs/Ancestors_molecular_function.tsv']
		d['GOAncestorsBP'] = ['GeneOntologyAggs/Ancestors_biological_process.tsv']
		d['GOAncestorsCC'] = ['GeneOntologyAggs/Ancestors_cellular_component.tsv']
		d['GOLookup'] = ['GeneOntologyAggs/SS_Lookup.tsv']
		d['GOTermLst'] = ['GeneOntologyAggs/GOTermLst.tsv']
		d['ProteinMapping'] = ['ProteinMapping.tsv']
		
		if 'featDict' in self.hyperParams:
			for item in self.hyperParams.featDict:
				d[item] = self.hyperParams.featDict[item]
		self.featDict = d
		self.modelType = None
		self.prioritizeDepth = self.hyperParams.get('prioritizeDepth',prioritizeDepth)
		
	def calcULCA(self,p1,p2):
		lst = []
		p1 = str(p1)
		p2 = str(p2)
		if p1 not in self.protMap or p2 not in self.protMap:
			for item in self.ancestors:
				lst.append(np.asarray([0]*self.ancestors[item].shape[0]))
			return np.hstack(lst)
		for ns in self.ancestors:
			#get GO terms per protein
			terms1 = self.protMap[p1][ns]
			terms2 = self.protMap[p2][ns]
			if terms1.shape[0] == 0 or terms2.shape[0] == 0:
				lst.append([0]*self.ancestors[ns].shape[1])
				continue
#			print(ns,terms1,terms2)
			#get ancestors of p1
			p1a = self.ancestors[ns][terms1].sum(axis=0)
#			print(np.where(p1a>0))
#			print('p1a',np.sum(p1a))
			#get ancestors of p2
			p2a = self.ancestors[ns][terms2].sum(axis=0)
#			print(np.where(p2a>0))
#			print('p2a',np.sum(p2a))
			#get common ancestors
			p1a = np.asarray(p1a).squeeze()
			p2a = np.asarray(p2a).squeeze()
			pa = p1a * p2a
#			print(np.where(pa>0))
#			print('pa1',np.sum(pa))
			#convert to binary
			pa[pa>0] = 1
#			print(np.where(pa>0))
#			print('pa2',np.sum(pa))
			#find lowest ancestor
			
			
			lca = np.argmax(pa*self.scoringData[ns])
#			print('lca',lca)
			
			#get descendents of lca
			lcaD = self.ancestors[ns].T[lca].todense()
			lcaD = np.asarray(lcaD).squeeze()
#			print(np.where(lcaD>0))
#			print('lcaD',np.sum(lcaD))
			#get ULCA
			ulca = lcaD * pa
#			print(np.where(ulca>0))
#			print('ucla',np.sum(ulca))
#			if np.sum(ulca) == 0:
#				exit(42)
			lst.append(ulca)
		return np.hstack(lst)
			
		
		
	def loadFeatureData(self,featureFolder):
		if featureFolder[-1] not in ['/','\\']:
			featureFolder+='/'
		
		#load ancestors for each ontology
		self.ancestors = {}
		self.scoringData = {}
		for item in ['MF','BP','CC']:
			if 'GOAncestors'+item in self.featDict:
				self.ancestors[item] = csr_matrix(np.asarray(PPIPUtils.parseTSV(featureFolder+self.featDict['GOAncestors'+item][0],'int')))
				self.scoringData[item] = np.zeros(self.ancestors[item].shape[0])


		#load scoring and indexing for each go term
		lookupData = PPIPUtils.parseTSV(featureFolder+self.featDict['GOLookup'][0])
		header = {}
		for item in lookupData[0]:
			header[item] = len(header)
		termToIdx = {}
		
		nsLookup = {'biological_process':'BP','molecular_function':'MF','cellular_component':'CC'}
		for line in lookupData[1:]:
			ns = line[header['Namespace']]
			ns = nsLookup.get(ns,ns)
			term = line[header['GO Name']]
			idx = int(line[header['LookupIDX']])
			icVal = float(line[header['IC Val']])
			depth = int(line[header['Depth']])
			self.scoringData[ns][idx] = (depth+icVal/1000 if self.prioritizeDepth else icVal+depth/10000) + 1 #+1 to ensure all vals > 0
			termToIdx[term] = idx
			

		
		#map each go term to uniprot
		termData = PPIPUtils.parseTSV(featureFolder+self.featDict['GOTermLst'][0])
		
		uniToGO = {}
		header=  {}
		for item in termData[0]:
			header[item] = len(header)
		for line in termData[1:]:
			uni = line[header['UniprotName']]
			ns = line[header['Branch']]
			term = line[header['Term']]
			if uni not in uniToGO:
				uniToGO[uni] = {}
				for item in self.ancestors:
					uniToGO[uni][item] = set()
			uniToGO[uni][ns].add(termToIdx[term])
		
		#map from proteins to ids, if needed.  Otherwise, just use uniprot map
		if self.featDict['ProteinMapping'][0] is not None:
			self.protMap = {}
			proteinMapping = PPIPUtils.parseTSV(featureFolder+self.featDict['ProteinMapping'][0])
			for line in proteinMapping:
				if line[0] not in self.protMap:
					self.protMap[line[0]] = {}
					for item in self.ancestors:
						self.protMap[line[0]][item] = set()
				if line[1] in uniToGO:
					for item in self.ancestors:
						self.protMap[line[0]][item] |= uniToGO[line[1]][item]
		else:
			self.protMap = uniToGO
				
		#convert sets to numpy arrays
		for item in self.protMap:
			for ns in self.protMap[item]:
				self.protMap[item][ns] = np.asarray(list(self.protMap[item][ns]))
			
			
	def genFeatureData(self,pairs,dataType='train',returnDict=False):
		classData = pairs[:,2]
		classData = classData.astype(np.int)
		pairs = pairs[:,0:2]
		retVals = []
		for i in range(0,pairs.shape[0]):
			retVals.append(self.calcULCA(pairs[i][0],pairs[i][1]))
			if i % 1000 == 0:
				print(i)
		print(len(pairs),np.sum(retVals))
		return np.vstack(retVals), classData

	def predictPairs(self, testPairs):
		if len(testPairs) > 100:
			return self.predictFromBatch(testPairs,64)
		else:
			return super().predictPairs(testPairs)

#	def fit(self,trainFeatures,trainClasses):
#		self.genModel() #create a new model from scratch, ensuring we don't overwrite the previously trained one
#		x = np.sum(trainFeatures,axis=1)
#		newFeats = trainFeatures[x!=0]
#		newClasses = trainClasses[x!=0]
#		print(newFeats.shape,trainFeatures.shape)
#		self.model.fit(newFeats,newClasses)

#	def predict_proba(self,predictFeatures,predictClasses):
#		x = np.sum(predictFeatures,axis=1)
#		newPredict = predictFeatures[x!=0]
#		preds = self.model.predict_proba(newPredict)
#		finalPreds = np.zeros((predictClasses.shape[0],2))
#		finalPreds[x==0] = (1,-1)
#		finalPreds[x!=0] = preds
#		return (finalPreds,np.asarray(predictClasses,dtype=np.int))
		
		
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
