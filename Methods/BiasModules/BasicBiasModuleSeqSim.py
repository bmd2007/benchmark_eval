#Based on paper Using support vector machine combined with auto covariance to predict protein–protein interactions from protein sequences by Yanzhi Guo, Lezheng Yu, Zhining Wen, and Menglong Li

import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
import math
import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
from GenericModule import GenericModule
from GenericProteinScorer import GenericProteinScorer 
from collections import Counter
import heapq
from BasicBiasModule import BasicBiasModule
class BasicBiasModuleSeqSim(BasicBiasModule):
	def __init__(self, hyperParams = None,maxNeighbors=10):
		GenericModule.__init__(self,hyperParams)
		self.featDict = self.hyperParams.get('featDict',{'SeqSim':['all-vs-all.tsv']})
		self.maxNeighbors = self.hyperParams.get('maxNeighbors',maxNeighbors)
		self.fullLst = []
		self.totalPos = 0
	def genModel(self,scoreData,mappingData):
		self.model = GenericProteinScorer(scoreData,mappingData)

	def fit(self,trainFeatures,trainClasses):
		scoreData = Counter()
		mappingData = {}
		for i in range(0,trainFeatures.shape[0]):
			curClass = trainClasses[i]*2-1 #convert to +1/-1
			for j in range(0,2):
				scoreData[str(trainFeatures[i,j])] += curClass
				
		sorts = {}
		f = open(self.featureFolder+self.featDict['SeqSim'][0])
		header = {'prot1':0,'prot2':1,'ident':2,'align_length':3,'mismatches':4,'gaps':5,'start_q':6,'stop_q':7,'start_s':8,'stop_s':9,'eval':10,'bit_score':11}
		for line in f:
			line = line.strip().split()
			p1 = line[header['prot1']]
			p2 = line[header['prot2']]
			ev = float(line[header['eval']])
			if ev == 0:
				ev = 1
			else:
				ev = min(abs(math.log(ev,10)),200)/200

			if p1 > p2:
				continue #skip reverse mappings, since all mappings appear twice in file
			
			if p1 not in sorts:
				sorts[p1] = []
			if p2 not in sorts:
				sorts[p2] = []
			
			if p2 in scoreData: #if p2 is in training data
				heapq.heappush(sorts[p1],(ev,p2))
				if len(sorts[p1]) > self.maxNeighbors: #remove lowest match
					x = heapq.heappop(sorts[p1])

			if p1 == p2:
				continue
			if p1 in scoreData: #if p1 is in trainingData
				heapq.heappush(sorts[p2],(ev,p1))
				if len(sorts[p2]) > self.maxNeighbors: #remove lowest match
					x = heapq.heappop(sorts[p2])
			
		f.close()
		
		for item,vals in sorts.items():
			weightSum = 0
			for i in range(0,len(vals)):
				weightSum += vals[i][0]
			if weightSum == 0:
				continue #no weigth should be impossible if matches are found.
				
			mappingData[item] = []
			for i in range(0,len(vals)):
				mappingData[item].append((vals[i][1],vals[i][0]/weightSum)) #append name, weight.  Divide by weight sum to make weights add up to 1
				
		self.genModel(scoreData,mappingData) #create a new model from scratch, ensuring we don't overwrite the previously trained one
		
		
	def predictPairs(self, testPairs):
		testFeatures, testClasses = self.genFeatureData(testPairs,'predict')
		testFeatures = self.scaleFeatures(testFeatures,'test')
		results = self.predict_proba(testFeatures,testClasses)
		z = results[0]
		testClasses = results[1]
		lst = []
		curPos = 0
		for i in range(0,z.shape[0]):
			pScores = []
			for idx in range(0,2):
				total = 0
				prot = str(testPairs[i,idx])
				if prot not in self.model.protMap:
					pScores.append(0)
					continue #0 score
				for (mappedProt,weight) in self.model.protMap[prot]:
					if mappedProt not in self.model.protMap:
						continue #0 score
					total += self.model.protScores[mappedProt] * weight
				pScores.append(total)
			lst.append((testPairs[i,0],testPairs[i,1],z[i,1],pScores[0],pScores[1],testClasses[i]))
			curPos += testClasses[i]
		
		self.totalPos += curPos
		print('cp',curPos)
		lst.sort(key=lambda x: -x[2])
		for i in range(0,10):
			print(i,lst[i])
		
		for i in range(0,500):
			self.fullLst.append(lst[i])
			
		print('\n\n\n')
		print('tot p',self.totalPos)
		self.fullLst.sort(key=lambda x: -x[2])
		for i in range(0,30):
			print(i,self.fullLst[i])
		
		return results
