import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import time
import numpy as np
from joblib import dump, load
import torch
import copy
from ProteinFeaturesHolder import ProteinFeaturesHolder
from GenericModule import GenericModule
class GenericPairwisePredictor(GenericModule):
	def __init__(self, hyperParams = None):
		if hyperParams is None:
			hyperParams = {}
		self.hyperParams = copy.deepcopy(hyperParams)
		
		if 'featScaleClass' not in self.hyperParams:
			self.featScaleClass = None
		else:
			self.featScaleClass = self.hyperParams['featScaleClass']
		
		self.featDict=  None
		self.TrainFiles = self.hyperParams.get('TrainFiles',[])
		self.TestFiles = self.hyperParams.get('TestFiles',[])
		self.ColumnNames = self.hyperParams.get('Columns',[])
		self.AugmentFunctions = self.hyperParams.get('Augment',[])
		self.testDataset = self.hyperParams.get('testData','test.tsv')
		self.trainDataset = self.hyperParams.get('trainData','train.tsv')
		self.datasetHeaders = self.hyperParams.get('datasetHeaders',False)
		self.replaceMissing = self.hyperParams.get('replaceMissing',0)
		
		if 'seed' in hyperParams:
			self.seed = int(hyperParams['seed'])
		else:
			self.seed = 1
		
		self.model=None
		self.scaleData = None
		self.featureFolder = None
	
	#load from each listed file to each key in the featDict
	def loadFeatureData(self,featureFolder):
		self.featureFolder =featureFolder
		
	
	def genFeatureData(self,pairs,dataType='train',returnDict=False):
		return None

	def augment(self,features):
		features = np.asarray(features)
		if len(self.AugmentFunctions) == features.shape[1]:
			for i in range(0,len(self.AugmentFunctions)):
				aug = self.AugmentFunctions[i]
				missing = np.isnan(features[i,:])
				nonmssing = missing==False
				if aug == 'log':
					features[i,nonmssing] = np.log(features[i,nonmssing]+1e-8)
				elif aug == 'log10':
					features[i,nonmssing] = np.log10(features[i,nonmssing]+1e-8)
				elif aug == 'log2':
					features[i,nonmssing] = np.log2(features[i,nonmssing]+1e-8)
				elif aug == 'sqrt':
					features[i,nonmssing] = np.sqrt(np.abs(features[i,nonmssing])) * np.sign(features[i,nonmissing])
				
				features[i,missing] = self.replaceMissing
		else:
			features[np.isnan(features)] = self.replaceMissing
		return features
	
	def train(self,trainPairs=None):
		for (trainFeatures, trainClasses) in self.loadDataFromFiles(trainPairs,'train'):
			trainFeatures = self.augment(trainFeatures)
			trainClasses = np.asarray(trainClasses)
			trainFeatures = self.scaleFeatures(trainFeatures,'train')
			self.fit(trainFeatures,trainClasses)
		
	def fit(self,trainFeatures,trainClasses):
		self.genModel() #create a new model from scratch, ensuring we don't overwrite the previously trained one
		self.model.fit(trainFeatures,trainClasses)


	#Predict using Full Data Method.  Assumes all data has been loaded to created test features, and that entire test dataset and features fit into memory
	def predictPairs(self, testPairs=None):
		return self.predictPairsFromBatch(testPairs,64)
	
	def predictPairsFromBatch(self,testPairs,batchSize=64):
		predictionsLst = []
		predictClassesLst = []
		for (predictFeatures, predictClasses) in self.loadDataFromFiles(testPairs,'predict',batchSize):
			predictFeatures = self.augment(predictFeatures)
			predictFeatures = self.scaleFeatures(predictFeatures,'test')
			p,c = self.predict_proba(predictFeatures,predictClasses)
			predictionsLst.append(p)
			predictClassesLst.append(c)
		return (np.vstack(predictionsLst),np.hstack(predictClassesLst))
	
		
	def loadDataFromFiles(self,pairs,fileType,batchSize=None,delim='\t'):
		#filters out only the rows from the files containing the pairs we need
		if pairs is not None:
			self.pairSet = set()
			for item in pairs:
				self.pairSet.add(tuple(item))
		else:
			self.pairSet = None
				
		if fileType == 'train':
			fLst = self.TrainFiles
			datasetFile = self.trainDataset
		elif fileType == 'predict' or fileType == 'test':
			fLst =  self.TestFiles
			datasetFile = self.testDataset
		self.curFiles = []
		for item in fLst:
			self.curFiles.append(open(self.featureFolder+item))
		self.datasetFile = open(self.featureFolder+datasetFile)
		self.curFilesHeader= []
		for item in self.curFiles:
			self.curFilesHeader.extend(item.readline().strip().split(delim))
		
		self.curHeaderDict = {}
		idx = 0
		for item in self.curFilesHeader:
			if item not in self.curHeaderDict:
				self.curHeaderDict[item] = idx
			idx += 1
		
		if self.datasetHeaders:
			dHeader = self.datasetFile.readline()
		
		fail =False
		for item in self.ColumnNames:
			if item not in self.curHeaderDict:
				print('Error, missing column name: ', item)
				fail = True
		if fail:
			exit(42)
		
		classData = []
		featureData = []
		curLine = self.curFilesHeader
		#parseFiles
		while len(curLine) > 0:
			curLine = []
			for item in self.curFiles:
				curLine.extend(item.readline().strip().split(delim))
			if len(curLine[0]) == 0:
				break #no more features
			proteinData = self.datasetFile.readline().strip().split(delim)
			
			p1, p2, c = proteinData
			
			classData.append(int(c))
			curData = []
			for item in self.ColumnNames:
				val = curLine[self.curHeaderDict[item]]
				if val == '?':
					val = np.nan
				else:
					try:
						val = float(val)
					except:
						pass
				curData.append(val)
			featureData.append(curData)
			if self.pairSet is not None:
				if (p1,p2) not in self.pairSet:
					continue #not one of the pairs we are parsing
			
			
			if batchSize is not None and len(classData) == batchSize:
				yield(featureData,classData)
				featureData = []
				classData = []
			
		if len(classData) > 0:
			yield (featureData,classData)
				
			
		
	#Predict using batches method.   Assumes all data has been loaded, but computing features for all pairs in memory at once would be infeasible.
	def predictFromBatch(self,testPairs,batchSize):
		pass
		
	def predict_proba(self,predictFeatures,predictClasses):
		preds = self.model.predict_proba(predictFeatures)
		return (preds,np.asarray(predictClasses,dtype=np.int))
		
	
	def parseTxtGenerator(self,tsvFile,batch,sep='\t',headerLines=1,classIdx = -1):
		f = open(tsvFile)
		header = None
		curData = []
		classData = []
		for line in f:
			if headerLines >0:
				if header is None:
					header = line.strip().split(sep)
				else:
					header = [header]
					header.append(line.strip().split(sep))
				headerLines -=1
				continue
			line = line.strip().split(sep)
			if classIdx == -1:
				classIdx = len(line)-1
			classData.append(int(line[classIdx]))
			line = line[:classIdx] + line[(classIdx+1):]
			line = [float(s) for s in line]
			curData.append(line)
			
			if len(curData) == batch:
				yield (header,curData,classData)
				curData =[]
				classData = []
		if len(curData) > 1:
			yield (header,curData,classData)
			curData = []
			classData = []
