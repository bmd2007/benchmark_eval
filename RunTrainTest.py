import numpy as np
import PPIPUtils
import time


def writePredictions(fname,predictions,classes):
	f = open(fname,'w')
	for i in range(0,predictions.shape[0]):
		f.write(str(predictions[i])+'\t'+str(classes[i])+'\n')
	f.close()

def writeScore(predictions,classes, fOut, predictionsFName=None, thresholds=[0.01,0.03,0.05,0.1,0.25,0.5,1]):
	#concate results from each fold, and get total scoring metrics
	finalPredictions = np.hstack(predictions)
	finalClasses = np.hstack(classes)
	results = PPIPUtils.calcScores(finalClasses,finalPredictions,thresholds)
	
	#format total metrics, and write them to a file
	lst = PPIPUtils.formatScores(results,'Total')
	for line in lst:
		fOut.write('\t'.join(str(s) for s in line) + '\n')
		print(line)
	
	fOut.write('\n')
	
	if predictionsFName is not None:
		writePredictions(predictionsFName,finalPredictions,finalClasses)



#modelClass - the class of the model to use
#outResultsName -- Name of File to write results to
#trainSets -- lists of protein pairs to use in trainings (p1, p2, class)
#testSets  -- lists of protein pairs to use in testings (p1, p2, class)
#featureFolder -- Folder containing dataset/features for classifier to pull features from
#hyperparams -- Any additional hyperparameters to pass into the model
#loadedModel -- Contains model already loaded with features, useful when combined with modelsLst argument to use different test sets on trained models
#modelsLst -- Optional argument containing list of already training models.  If provided, these models can be used in place of training (note:  argument is designed to take the list of models that this function returns so they can be used on a different test set)
def runTest(modelClass, outResultsName,trainSets,testSets,featureFolder,hyperParams = {},predictionsName =None,loadedModel= None,modelsLst = None,thresholds = [0.01,0.03,0.05,0.1,0.25,0.5,1],resultsAppend=False,keepModels=False,saveModels=None,predictionsFLst = None, startIdx=0,loads=None):
	#record total time for computation
	t = time.time()
	
	if featureFolder[-1] != '/':
		featureFolder += '/'
	
	#open file to write results to for each fold/split
	if resultsAppend and outResultsName:
		outResults = open(outResultsName,'a')
	elif outResultsName:
		outResults = open(outResultsName,'w')
	#keep list of predictions/classes per fold
	totalPredictions = []
	totalClasses = []
	trainedModelsLst = []

	for i in range(0,startIdx):
		totalPredictions.append([])
		totalClasses.append([])
		trainedModelsLst.append([])
	
	#create the model once, loading all the features and hyperparameters as necessary
	if loadedModel is not None:
		model = loadedModel
	else:
		model = modelClass(hyperParams)
		model.loadFeatureData(featureFolder)
	
	for i in range(startIdx,len(testSets)):
		model.batchIdx = i

		#create model, passing training data, testing data, and hyperparameters
		if modelsLst is None:
			model.batchIdx = i
			if loads is None or loads[i] is None:
				#run training and testing, get results
				model.train(trainSets[i])
				if saveModels is not None:
					print('save')
					model.saveModelToFile(saveModels[i])
			else:
				model.loadModelFromFile(loads[i])
				#model.setScaleFeatures(trainSets[i])

			preds, classes = model.predictPairs(testSets[i])
			if keepModels:
				trainedModelsLst.append(model.getModel())	
		else:
			#if we are given a model, skip the training and use it for testing
			preds, classes = model.predictPairs(testSets[i],modelLst[i])
			if keepModels:
				trainedModelsLst.append(modelLst[i])
		
		
		print('pred')
		#compute result metrics, such as Max Accuracy, Precision/Recall @ Max Accuracy, Average Precition, and Max Precition @ k
		results = PPIPUtils.calcScores(classes,preds[:,1],thresholds)
		#format the scoring results, with line for title
		lst = PPIPUtils.formatScores(results,'Fold '+str(i))
		#write formatted results to file, and print result to command line
		if outResultsName:
			for line in lst:
				outResults.write('\t'.join(str(s) for s in line) + '\n')
				print(line)
			outResults.write('\n')
		else:
			for line in lst:
				print(line)
		print(time.time()-t)
		
		#append results to total results for overall scoring
		totalPredictions.append(preds[:,1])
		totalClasses.append(classes)

		if predictionsFLst is not None:
			writePredictions(predictionsFLst[i],totalPredictions[i],totalClasses[i])
			
	if not resultsAppend and predictionsName is not None and outResultsName: #not appending. calculate total results
		writeScore(totalPredictions,totalClasses,outResults,predictionsName,thresholds)
	
	#output the total time to run this algorithm
	if outResultsName:
		outResults.write('Time: '+str(time.time()-t))
		outResults.close()
	return (totalPredictions, totalClasses,model,trainedModelsLst)
	
	
	
	
	
	
	
	
	
#Same as runTest, but takes a list of list of tests, and a list of filenames, allowing running multiple test sets and storing to multiple files
def runTestLst(modelClass, outResultsNameLst,trainSets,testSetsLst,featureFolder,hyperParams = {},predictionsNameLst=None,loadedModel= None,modelsLst = None,thresholds = [0.01,0.03,0.05,0.1,0.25,0.5,1],resultsAppend=False,keepModels=False,saveModels=None,predictionsFLst = None,startIdx=0,loads=None):
	#record total time for computation
	t = time.time()
	if featureFolder[-1] != '/':
		featureFolder += '/'
	
	#keep list of predictions/classes per fold
	totalPredictions = []
	totalClasses = []
	trainedModelsLst = []
	
	#open file to write results to for each fold/split
	outResults = []
	
	for i in range(0,len(testSetsLst)):
		if outResultsNameLst is not None:
			if resultsAppend:
				outResults.append(open(outResultsNameLst[i],'a'))
			else:
				outResults.append(open(outResultsNameLst[i],'w'))
		totalPredictions.append([])
		totalClasses.append([])
		trainedModelsLst.append([])
		
	for i in range(0,startIdx):
		for j in range(0,len(totalPredictions)):
			totalPredictions[j].append([])
			totalClasses[j].append([])
			trainedModelsLst[j].append([])
	
	#create the model once, loading all the features and hyperparameters as necessary
	if loadedModel is not None:
		model = loadedModel
	else:
		model = modelClass(hyperParams)
		model.loadFeatureData(featureFolder)
	
	for i in range(startIdx,len(testSetsLst[0])):
		model.batchIdx = i

		#create model, passing training data, testing data, and hyperparameters
		if modelsLst is None:
			model.batchIdx = i
			
			if loads is None or loads[i] is None:
				#run training and testing, get results
				model.train(trainSets[i])
				if saveModels is not None:
					print('save')
					model.saveModelToFile(saveModels[i])
			else:
				print('load')
				model.loadModelFromFile(loads[i])
				#model.setScaleFeatures(trainSets[i])
			if keepModels:
				trainedModelsLst.append(model.getModel())	
			
		else:
			#if we are given a model, skip the training and use it for testing
			if keepModels:
				trainedModelsLst.append(modelLst[i])
		
		for testIdx in range(0,len(testSetsLst)):
			if modelsLst is None:
				preds, classes = model.predictPairs(testSetsLst[testIdx][i])
			else:
				preds, classes = model.predictPairs(testSetsLst[testIdx][i],modelLst[i])
			
			#compute result metrics, such as Max Accuracy, Precision/Recall @ Max Accuracy, Average Precition, and Max Precition @ k
			results = PPIPUtils.calcScores(classes,preds[:,1],thresholds)
			#format the scoring results, with line for title
			lst = PPIPUtils.formatScores(results,'Fold '+str(i))
			#write formatted results to file, and print result to command line
			if outResultsNameLst is not None:
				for line in lst:
					outResults[testIdx].write('\t'.join(str(s) for s in line) + '\n')
					print(line)
				outResults[testIdx].write('\n')
			else:
				for line in lst:
					print(line)
			print(time.time()-t)
			
			#append results to total results for overall scoring
			totalPredictions[testIdx].append(preds[:,1])
			totalClasses[testIdx].append(classes)
			if predictionsFLst is not None:
				writePredictions(predictionsFLst[testIdx][i],totalPredictions[testIdx][i],totalClasses[testIdx][i])

	if outResultsNameLst is not None:
		for testIdx in range(0,len(testSetsLst)):
			if not resultsAppend: #not appending. calculate total results
				writeScore(totalPredictions[testIdx],totalClasses[testIdx],outResults[testIdx],(predictionsNameLst[testIdx] if predictionsNameLst is not None else None),thresholds)
			
			#output the total time to run this algorithm
			outResults[testIdx].write('Time: '+str(time.time()-t))
			outResults[testIdx].close()
	return (totalPredictions, totalClasses,model,trainedModelsLst)
	
	
	


#Same as runTest, but takes a list of list of tests, and a list of filenames, allowing running multiple test sets and storing to multiple files
def runTestPairwiseFoldersLst(modelClass, outResultsName,trainSets,testSets,featureFolderLst,hyperParams = {},predictionsName=None,loadedModel= None,modelsLst = None,thresholds = [0.01,0.03,0.05,0.1,0.25,0.5,1],resultsAppend=False,keepModels=False,saveModels=None,predictionsFLst = None,startIdx=0,loads=None):
			
	#record total time for computation
	t = time.time()
	for i in range(0,len(featureFolderLst)):
		if featureFolderLst[i][-1] != '/':
			featureFolderLst[i] += '/'
	
	
	#open file to write results to for each fold/split
	if resultsAppend and outResultsName:
		outResults = open(outResultsName,'a')
	elif outResultsName:
		outResults = open(outResultsName,'w')
	#keep list of predictions/classes per fold
	totalPredictions = []
	totalClasses = []
	trainedModelsLst = []

	for i in range(0,startIdx):
		totalPredictions.append([])
		totalClasses.append([])
		trainedModelsLst.append([])
	
	#create the model once, loading all the features and hyperparameters as necessary
	if loadedModel is not None:
		model = loadedModel
	else:
		model = modelClass(hyperParams)
	
	for i in range(startIdx,len(featureFolderLst)):
		#need to use new folder per test
		model.loadFeatureData(featureFolderLst[i])
		model.batchIdx = i
		
		#create model, passing training data, testing data, and hyperparameters
		if modelsLst is None:
			model.batchIdx = i
			if loads is None or loads[i] is None:
				#run training and testing, get results
				model.train(trainSets[i] if trainSets is not None else None)
				if saveModels is not None:
					print('save')
					model.saveModelToFile(saveModels[i])
			else:
				model.loadModelFromFile(loads[i])
				#model.setScaleFeatures(trainSets[i])

			preds, classes = model.predictPairs(testSets[i] if testSets is not None else None)
			if keepModels:
				trainedModelsLst.append(model.getModel())	
		else:
			#if we are given a model, skip the training and use it for testing
			model.setModel(modelsLst[i])
			preds, classes = model.predictPairs(testSets[i] if testSets is not None else None)
			if keepModels:
				trainedModelsLst.append(modelLst[i])
		
		
		print('pred')
		#compute result metrics, such as Max Accuracy, Precision/Recall @ Max Accuracy, Average Precition, and Max Precition @ k
		results = PPIPUtils.calcScores(classes,preds[:,1],thresholds)
		#format the scoring results, with line for title
		lst = PPIPUtils.formatScores(results,'Fold '+str(i))
		#write formatted results to file, and print result to command line
		if outResultsName:
			for line in lst:
				outResults.write('\t'.join(str(s) for s in line) + '\n')
				print(line)
			outResults.write('\n')
		else:
			for line in lst:
				print(line)
		print(time.time()-t)
		
		#append results to total results for overall scoring
		totalPredictions.append(preds[:,1])
		totalClasses.append(classes)

		if predictionsFLst is not None:
			writePredictions(predictionsFLst[i],totalPredictions[i],totalClasses[i])
			
	if not resultsAppend and predictionsName is not None and outResultsName: #not appending. calculate total results
		writeScore(totalPredictions,totalClasses,outResults,predictionsName,thresholds)
	
	#output the total time to run this algorithm
	if outResultsName:
		outResults.write('Time: '+str(time.time()-t))
		outResults.close()
	return (totalPredictions, totalClasses,model,trainedModelsLst)
	
	
	

