import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import PPIPUtils
import PairwisePreprocessUtils
import numpy as np
from scipy.sparse import csr_matrix

from collections import Counter

class GenericTermPairScorer(object):
	def __init__(self,termPairLst,interactionLst,prefix,selfInteractions=False,headerLines=1):
		#whether to allow comparisons between a protein and itself
		self.selfInteractions=selfInteractions
		self.prefix = prefix
		
		#process all terms, and map terms to indices
		self.allTerms = {}
		self.termCounts = []
		self.proteinTerms = {}
		for line in termPairLst:
			if headerLines > 0:
				headerLines-=1
				continue
			prot = line[0]
			term = line[1]
			if prot not in self.proteinTerms:
				self.proteinTerms[prot] = set()
			if term not in self.allTerms:
				self.allTerms[term] = len(self.allTerms)
				self.termCounts.append(0)
			termIdx = self.allTerms[term]
			self.termCounts[termIdx]+=1
			self.proteinTerms[prot].add(termIdx)
		
		
		#create a sparse matrix of interacting terms based on the interactionLst and termIdxs
		interactions = Counter()
		for (prot1, prot2) in interactionLst:
			if prot1 not in self.proteinTerms or prot2 not in self.proteinTerms: #one of the proteins has no terms, skip interaction
				continue
			for term1 in self.proteinTerms[prot1]:
				for term2 in self.proteinTerms[prot2]:
					interactions[(term1,term2)] += 1
					interactions[(term2,term1)] += 1
		
		indicesLst = [[],[]]
		valuesLst = []
		for ((t1,t2),val) in interactions.items():
			indicesLst[0].append(t1)
			indicesLst[1].append(t2)
			valuesLst.append(val)
		
		#torch doesn't allow tensor based indexing into sparse matrices, switching to numpy
		#torch.sparse_coo_tensor(indicesLst,valuesLst,(len(self.allTerms),len(self.allTerms)),device=self.deviceType)
		self.termIntMatrix = csr_matrix((valuesLst,indicesLst),(len(self.allTerms),len(self.allTerms)))
		
		#create a torch tensor containing term counts
		#self.termCounts = torch.tensor(self.termCounts,device=self.deviceType)
		self.termCounts = np.asarray(self.termCounts)
		
		#create a torch tensor containing self-interactions between term pairs (term pairs on same protein)
		if self.selfInteractions:
			#if we are allowing selfInteractions, just create a matrix of zeros, since we are not filtering out term pairs in a single protein
			#self.protTermPairMat = torch.sparse_coo_tensor([[0],[0]],[0],(len(self.allTerms),len(self.allTerms)),device=self.deviceType)
			self.protTermPairMat = csr_matrix(([0],[[0],[0]]),(len(self.allTerms),len(self.allTerms)))
		else:
			termPairs = Counter()
			for prot in self.proteinTerms:
				lst = list(self.proteinTerms[prot])
				for i in range(0,len(lst)):
					termPairs[(lst[i],lst[i])]+=1
					for j in range(i+1,len(lst)):
						termPairs[(lst[i],lst[j])]+=1
						termPairs[(lst[j],lst[i])]+=1
			termIdxLst = [[],[]]
			termValLst = []
			for ((t1,t2),val) in termPairs.items():
				termIdxLst[0].append(t1)
				termIdxLst[1].append(t2)
				termValLst.append(val)
			#self.protTermPairMat = torch.sparse_coo_tensor(termIdxLst,termValLst,(len(self.allTerms),len(self.allTerms)),device=self.deviceType)
			self.protTermPairMat = csr_matrix((termValLst,termIdxLst),(len(self.allTerms),len(self.allTerms)))
		
		#convert dictionary containing term indices per protein to torch tensors
		for prot in self.proteinTerms:
			self.proteinTerms[prot] = np.asarray(sorted(self.proteinTerms[prot]))
			#self.proteinTerms[prot] = torch.tensor(sorted(self.proteinTerms[prot]),device=self.deviceType)
		
	def scoreProteinPair(self,id1,id2,dictionary={},prefix=None):
		if prefix is None:
			prefix = self.prefix
		
	
		emptySet = None
		#get goTermIdxs for each protein
		p1Idxs = self.proteinTerms.get(id1,emptySet)
		p2Idxs = self.proteinTerms.get(id2,emptySet)
		
		
		#do calculations 
		idxs1 = p1Idxs
		idxs2 = p2Idxs
			
		#no  terms for at least one protein
		if idxs1 is None or idxs2 is None:
			PairwisePreprocessUtils.matrixCalculationsNP(None,prefix,dictionary)
			return dictionary
				
		
		#get values
		#get counts of each term pair in interaction
		interactionCountsMat = PairwisePreprocessUtils.getMatFromMatrixNP(self.termIntMatrix,idxs1,idxs2)
			
		
		#get the count of the total number of possible protein pairs for each pair of terms to be involved in
		#allPairsCountsMat = self.termCounts[idxs1].unsqueeze(0).T + self.termCounts[idxs2]
		allPairsCountsMat = np.expand_dims(self.termCounts[idxs1],0).T * self.termCounts[idxs2]
			
		#get counts of each term within the same protein, to exclude from counting
		selfIntCountsMat = PairwisePreprocessUtils.getMatFromMatrixNP(self.protTermPairMat,idxs1,idxs2)
			
		#compute frequency calculated per term pair ((total interactions containing term pair)/(total protein pairs containing term pair))
		#note, if id1==id2 and self.selfInteractions is set to False, this can throw a divide by zero error
		freqMat = interactionCountsMat/(allPairsCountsMat-selfIntCountsMat)
		
		#do pairwise calculations
		PairwisePreprocessUtils.matrixCalculationsNP(freqMat,prefix,dictionary)
		
		return dictionary
			
			
			