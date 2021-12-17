import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import PPIPUtils
import PairwisePreprocessUtils
import numpy as np 
from collections import Counter
from scipy.sparse import csr_matrix
import time
import torch
#much faster with dense matrices than sparse matrices. . .
#term pair matrix is about 10% full, which is running 10x slower than dense matrices
#edit, above statement is for running on all terms.  Uncertain if still faster using only level 2 terms, but matrices are much smaller, so should be fine.
class GOTermPairScorer(object):
	def __init__(self,goTermFile,interactionLst,prefix ='GOL2Freq_',selfInteractions=False,deviceType='cpu'):
		#whether to allow comparisons between a protein and itself
		self.selfInteractions=selfInteractions
		self.deviceType=deviceType
		self.prefix = prefix
		t= time.time()
		
		#load key value list of all goTerms (containing protName, namespace, and term name)
		data = PPIPUtils.parseTSV(goTermFile)
		header = {}
		for i in range(0,len(data[0])):
			header[data[0][i]] = i
		data = data[1:]
		
		#load all terms, and map terms to indices
		self.allTerms = {}
		self.termCounts = []
		self.proteinTerms = {}
		for line in data:
			term = line[header['GO L2 Term']]
			prot = line[header['ProtName']]
			ns = line[header['Namespace']]
			if prot not in self.proteinTerms:
				self.proteinTerms[prot] = {'CC':set(),'MF':set(),'BP':set(),'ALL':set()}
			if term not in self.allTerms:
				self.allTerms[term] = len(self.allTerms)
				self.termCounts.append(0)
			termIdx = self.allTerms[term]
			self.termCounts[termIdx]+=1
			self.proteinTerms[prot][ns].add(termIdx)
			self.proteinTerms[prot]['ALL'].add(termIdx)
		
		for prot, nsLst in self.proteinTerms.items():
			for ns in nsLst:
				nsLst[ns] = torch.tensor(sorted(nsLst[ns]))
		
		
		#create a sparse matrix of interacting terms based on the interactionLst and termIdxs
		self.termIntMatrix=torch.zeros((len(self.termCounts),len(self.termCounts)),device=self.deviceType)
		for (prot1, prot2) in interactionLst:
			if prot1 not in self.proteinTerms or prot2 not in self.proteinTerms: #one of the proteins has no terms, skip interaction
				continue
			terms1 = self.proteinTerms[prot1]['ALL']
			terms2 = self.proteinTerms[prot2]['ALL']
			t1 = terms1.repeat(terms2.shape[0])
			t2 = terms2.repeat_interleave(terms1.shape[0])
			self.termIntMatrix[t1,t2]+=1
			self.termIntMatrix[t2,t1]+=1
		
		
		
		#create a torch tensor containing term counts
		self.termCounts = torch.tensor(self.termCounts,device=self.deviceType)
		#self.termCounts = np.asarray(self.termCounts)
		
		#create a torch tensor containing self-interactions between term pairs (term pairs on same protein)
		if self.selfInteractions:
			#if we are allowing selfInteractions, just create a matrix of zeros, since we are not filtering out term pairs in a single protein
		#	self.protTermPairMat = torch.sparse_coo_tensor([[0],[0]],[0],(len(self.allTerms),len(self.allTerms)),device=self.deviceType)
			#self.protTermPairMat = csr_matrix(([0],[[0],[0]]),(len(self.allTerms),len(self.allTerms)))
			self.protTermPairMat = torch.zeros(self.termIntMatrix.shape,device=deviceType)
		
		else:
			self.protTermPairMat = torch.zeros(self.termIntMatrix.shape,device=deviceType)
			for prot in self.proteinTerms:
				terms = self.proteinTerms[prot]['ALL']
				t1 = terms.repeat(terms.shape[0])
				t2 = terms.repeat_interleave(terms.shape[0])
				self.protTermPairMat[t1,t2] += 1
				self.protTermPairMat[terms,terms] -= 1 #don't count (x,x) pairs
			
	def scoreProteinPair(self,id1,id2,dictionary={},prefix=None):
		if prefix is None:
			prefix = self.prefix
		emptySet = {'CC':None,'BP':None,'MF':None,'ALL':None}
		#get goTermIdxs per ontology for each protein
		p1Idxs = self.proteinTerms.get(id1,emptySet)
		p2Idxs = self.proteinTerms.get(id2,emptySet)
		#do calculations per ontology
		for ns in emptySet:
			curNSprefix = prefix+ns+'_'
			idxs1 = p1Idxs[ns]
			idxs2 = p2Idxs[ns]
			
			#no GO terms for one ontology for at least one protein
			if idxs1 is None or idxs2 is None or len(idxs1) == 0 or len(idxs2) == 0:
				PairwisePreprocessUtils.matrixCalculationsNP(None,curNSprefix,dictionary)
				continue
				
			#get values
			#get counts of each term pair in interaction
			interactionCountsMat = PairwisePreprocessUtils.getMatFromMatrix(self.termIntMatrix,idxs1,idxs2)
			
			#get the count of the total number of possible protein pairs for each pair of terms to be involved in
			allPairsCountsMat = self.termCounts[idxs1].unsqueeze(0).T * self.termCounts[idxs2]
			
			#get counts of each term within the same protein, to exclude from counting
			selfIntCountsMat = PairwisePreprocessUtils.getMatFromMatrix(self.protTermPairMat,idxs1,idxs2)

			#compute frequency calculated per term pair ((total interactions containing term pair)/(total protein pairs containing term pair))
			#note, if id1==id2 and self.selfInteractions is set to False, this can throw a divide by zero error
			freqMat = interactionCountsMat/(allPairsCountsMat-selfIntCountsMat)

			if freqMat.max() > 1:
				print(freqMat)
				print(interactionCountsMat)
				print(allPairsCountsMat)
				print(selfIntCountsMat)
				exit(42)
			#do pairwise calculations
			PairwisePreprocessUtils.matrixCalculations(freqMat,curNSprefix,dictionary)

		return dictionary
			