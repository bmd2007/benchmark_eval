from joblib import dump, load
import numpy as np 
class GenericProteinScorer(object):

	#takes scores per protein, and map of protein importance to neighbors, as arguments
	def __init__(self,protScores,protMap):
		self.protScores = protScores
		self.protMap = protMap
		
	#no fitting
	def fit(self,trainFeatures,trainClasses):
		return self
				
	def predict_proba(self,testPairs):
		scoreLst = []
		for item in testPairs:
			total = 0
			for i in range(0,2):
				prot = str(item[i])
				if prot not in self.protMap:
					continue #0 score
				for (mappedProt,weight) in self.protMap[prot]:
					if mappedProt not in self.protMap:
						continue #0 score
					total += self.protScores[mappedProt] * weight
			scoreLst.append([-total,total]) #-total for negative class, won't add to 1 (not actual probability), but can still be used to sort
		scoreLst = np.asarray(scoreLst)
		return scoreLst
		
		
	def saveModelToFile(self,fname):
		if self.protScores is None:
			print('Error, no model to save')
			exit(42)
		dump((self.protScores,self.protMap),fname)

	def loadModelFromFile(self,fname):
		self.protScores, self.protMap = load(fname)