from GenericModule import GenericModule
import numpy as np

#designed for usage with neural network models using dictionary datasets
class GenericNetworkModule(GenericModule):
	def __init__(self, hyperParams = {}):
		GenericModule.__init__(self,hyperParams)
		self.dataLookup = None
		self.dataMatrix = None
		self.validationRate = hyperParams.get('ValidationRate',None)
		self.scaleData=None
		
		
	#when running fit or predict, pass in the dataMatrix
	def fit(self,trainFeatures,trainClasses):
		if self.validationRate is not None and self.validationRate > 0:
			newTrainFeat, newTrainClass, newValidFeat, newValidClass = self.splitFeatures(trainFeatures,trainClasses,self.validationRate)
			self.genModel() #create a new model from scratch, ensuring we don't overwrite the previously trained one
			self.model.fit(newTrainFeat,newTrainClass,self.dataMatrix, newValidFeat, newValidClass)
		else:
			self.genModel() #create a new model from scratch, ensuring we don't overwrite the previously trained one
			self.model.fit(trainFeatures,trainClasses,self.dataMatrix)


	#when running fit or predict, pass in the dataMatrix
	def predict_proba(self,predictFeatures,predictClasses):
		preds = self.model.predict_proba(predictFeatures,self.dataMatrix)
		return (preds,predictClasses)
	
	def loadFeatureData(self,featureFolder):
		pass
		
	#no scaling in this model
	def scaleFeatures(self,features,scaleType):
		return features
		
	def saveFeatScaler(self,fname):
		pass
			
	def loadFeatScaler(self,fname):
		pass

	#swap out pair data for their indices in the data matrix, and return them as features
	def genFeatureData(self,pairs,dataType='train'):	
		classData = np.asarray(pairs[:,2],dtype=np.int32)
		orgFeatsData = pairs[:,0:2]
		#replace all pair values with their matrix index value
		featsData = [self.dataLookup[str(a)] for a in orgFeatsData.flatten()]
		featsData = np.asarray(featsData).reshape(classData.shape[0],2)
		return featsData, classData
	
	
	def predictFromBatch(self,testPairs,batchSize,model=None):
		#no reason to use batches since pairwise data isn't created until dataloader
		return self.predictPairs(testPairs,model)
		
	#no reason to load pairs from file for this model
	def predictFromFile(self,testFile,batchSize,sep='\t',headerLines=1,classIdx=-1,model=None):
		pass
