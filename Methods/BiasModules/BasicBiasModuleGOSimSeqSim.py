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

import time
import numpy as np
from ProteinFeaturesHolder import ProteinFeaturesHolder
from GenericModule import GenericModule
from GenericProteinScorer import GenericProteinScorer 
from collections import Counter
import heapq
from BasicBiasModule import BasicBiasModule
import math
class BasicBiasModuleGOSimSeqSim(BasicBiasModule):
	def __init__(self, hyperParams = None,maxNeighbors=5,goMapBonus = 1):
		GenericModule.__init__(self,hyperParams)
		self.featDict = self.hyperParams.get('featDict',{'SeqSim':['all-vs-all.tsv'],'ProteinMapping':['ProteinMapping.tsv']})
		self.maxNeighbors = self.hyperParams.get('maxNeighbors',maxNeighbors)
		self.goMapBonus = self.hyperParams.get('goMapBonus',goMapBonus)
		
	def genModel(self,scoreData,mappingData):
		self.model = GenericProteinScorer(scoreData,mappingData)

	def fit(self,trainFeatures,trainClasses):
		scoreData = Counter()
		mappingData = {}
		for i in range(0,trainFeatures.shape[0]):
			curClass = trainClasses[i]*2-1 #convert to +1/-1
			for j in range(0,2):
				scoreData[str(trainFeatures[i,j])] += curClass
		
		
		#get all id to protein mappings
		f = open(self.featureFolder+self.featDict['ProteinMapping'][0])
		protMap = {}
		revProtMap = {}
		for line in f:
			line = line.strip().split()
			if line[0] not in protMap:
				protMap[line[0]] = set()
			protMap[line[0]].add(line[1])
			if line[1] not in revProtMap:
				revProtMap[line[1]] = set()
			revProtMap[line[1]].add(line[0])
				
		f.close()
		
		#calculate all values from go mapping
		#for all mappings that share a proteins, calculate their intersection/union score
		goMapScoring = {}
		for item,prots in revProtMap.items():
			if len(prots) > 0:
				protLst = list(prots)
				for i in range(0,len(protLst)):
					for j in range(i+1,len(protLst)):
						intersect = len(protMap[protLst[i]] & protMap[protLst[j]])
						union = len(protMap[protLst[i]] | protMap[protLst[j]])
						goMapScoring[(min(protLst[i],protLst[j]),max(protLst[i],protLst[j]))] = intersect/union
		
		#calculate the eval of all similar sequences, add in goMapBonus + goMapScoring when pair is in goMapScoring
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
				
			#add scoring, if exists, and delete pair from mapping
			if (p1,p2) in goMapScoring:
				ev += goMapScoring[(p1,p2)] + self.goMapBonus
				del goMapScoring[(p1,p2)]
				
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
		for item,val in goMapScoring.items(): #process pairs in go map scoring that remain (had no sequence similarity)
			p1 = item[0]
			p2 = item[1]
			ev = val + self.goMapBonus
			if p1 not in sorts:
				sorts[p1] = []
			if p2 not in sorts:
				sorts[p2] = []
			
			if p2 in scoreData: #if p2 is in training data
				heapq.heappush(sorts[p1],(ev,p2))
				if len(sorts[p1]) > self.maxNeighbors: #remove lowest match
					x = heapq.heappop(sorts[p1])
					
			if p1 in scoreData: #if p1 is in trainingData
				heapq.heappush(sorts[p2],(ev,p1))
				if len(sorts[p2]) > self.maxNeighbors: #remove lowest match
					x = heapq.heappop(sorts[p2])
					
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
		
		
