import PPIPUtils
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
currentDir = currentdir+'/'
import numpy as np

#if dirLst, just get directory names, not data sets.  Used for pairwise predictors
def loadHumanRandom50(directory,augment = False,dirLst = False):
	trainSets = []
	testSets = []
	saves = []
	predFNames = []
	for i in range(0,5):
		if not dirLst:
			trainSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random50/Train_'+str(i)+'.tsv','int')))
			testSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random50/Test_'+str(i)+'.tsv','int')))
		saves.append(directory+'R50_'+str(i)+'.out')
		predFNames.append(directory+'R50_'+str(i)+'_predict.tsv')
		
	if augment:
		trainSets, testSets = augmentAll(trainSets,testSets)
	featDir = currentDir+'PPI_Datasets/Human2021/'
	if dirLst:
		featDir = []
		for i in range(0,5):
			featDir.append(currentDir+'PPI_Datasets/Human2021/BioGRID2021/pairwiseDatasets/r50_'+str(i)+'/')
	return trainSets,testSets, saves,predFNames, featDir

def loadHumanRandom20(directory,augment = False,dirLst = False):
	trainSets = []
	testSets = []
	testSets2 = []
	saves = []
	predFNames = [[],[]]
	for i in range(0,5):
		if not dirLst:
			trainSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random20/Train_'+str(i)+'.tsv','int')))
			testSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random20/Test1_'+str(i)+'.tsv','int')))
			testSets2.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random20/Test2_'+str(i)+'.tsv','int')))
		saves.append(directory+'R20_'+str(i)+'.out')
		predFNames[0].append(directory+'R20_'+str(i)+'_predict1.tsv')
		predFNames[1].append(directory+'R20_'+str(i)+'_predict2.tsv')
	if augment:
		trainSets, testSets = augmentAll(trainSets,testSets)
	featDir = currentDir+'PPI_Datasets/Human2021/'
	if dirLst:
		featDir = []
		newSaves = []
		for j in range(1,3):
			for i in range(0,5):
				featDir.append(currentDir+'PPI_Datasets/Human2021/BioGRID2021/pairwiseDatasets/r20_'+str(j)+'_'+str(i)+'/')
				name = saves[i].split('.')
				name = name[0]+'_'+str(j)+name[1]
				newSaves.append(name)
			
		predFNames = predFNames[0] + predFNames[1]
		saves = newSaves
	return trainSets,[testSets,testSets2], saves, predFNames, featDir


def loadHumanHeldOut50(directory,augment = False, dirLst=False):
	trainSets = []
	testSets = []
	saves = []
	predFNames = []
	for i in range(0,6):
		for j in range(i,6):
			if not dirLst:
				trainSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut50/Train_'+str(i)+'_'+str(j)+'.tsv','int')))
				testSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut50/Test_'+str(i)+'_'+str(j)+'.tsv','int')))
			saves.append(directory+'H50_'+str(i)+'_'+str(j)+'.out')
			predFNames.append(directory+'H50_'+str(i)+'_'+str(j)+'_predict.tsv')
	if augment:
		trainSets, testSets = augmentAll(trainSets,testSets)
	
	featDir = currentDir+'PPI_Datasets/Human2021/'
	if dirLst:
		featDir = []
		for i in range(0,6):
			for j in range(i,6):
				featDir.append(currentDir+'PPI_Datasets/Human2021/BioGRID2021/pairwiseDatasets/h50_'+str(i)+'_'+str(j)+'/')
	return trainSets,testSets, saves,predFNames, featDir

def loadHumanHeldOut20(directory,augment = False, dirLst = False):
	trainSets = []
	testSets = []
	testSets2 = []
	saves = []
	predFNames = [[],[]]
	for i in range(0,6):
		for j in range(i,6):
			if not dirLst:
				trainSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut20/Train_'+str(i)+'_'+str(j)+'.tsv','int')))
				testSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut20/Test1_'+str(i)+'_'+str(j)+'.tsv','int')))
				testSets2.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut20/Test2_'+str(i)+'_'+str(j)+'.tsv','int')))
			saves.append(directory+'H20_'+str(i)+'_'+str(j)+'.out')
			predFNames[0].append(directory+'H20_'+str(i)+'_'+str(j)+'_predict1.tsv')
			predFNames[1].append(directory+'H20_'+str(i)+'_'+str(j)+'_predict2.tsv')
	if augment:
		trainSets, testSets = augmentAll(trainSets,testSets)
	featDir = currentDir+'PPI_Datasets/Human2021/'
	if dirLst:
		featDir = []
		newSaves = []
		for k in range(1,3):
			idx = 0
			for i in range(0,6):
				for j in range(i,6):
					featDir.append(currentDir+'PPI_Datasets/Human2021/BioGRID2021/pairwiseDatasets/h20_'+str(k)+'_'+str(i)+'_'+str(j)+'/')
					name = saves[idx].split('.')
					name = name[0]+'_'+str(k)+name[1]
					newSaves.append(name)
					idx += 1
			
		predFNames = predFNames[0] + predFNames[1]
		saves = saves + saves
	return trainSets,[testSets,testSets2], saves,predFNames, featDir

def loadMartinHPylori(directory,kfolds=5):
	trainSets = []
	testSets = []
	saves = []
	predFNames = []
	#load pairs
	data = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Martin_H_pylori/allPairs.tsv','int')

	#create folds
	k = kfolds
	trainSets,testSets = PPIPUtils.createKFoldsAllData(data,k,seed=1)

	for i in range(0,k):
		saves.append(directory+'Martin_H_Pylori_'+str(i)+'.out')
		predFNames.append(directory+'Martin_H_Pylori_'+str(i)+'_predict.tsv')
	
	return trainSets,testSets, saves,predFNames, currentDir+'PPI_Datasets/Martin_H_pylori/'
	

def loadGuoYeastDataTian(directory,kfolds=5):
	trainSets = []
	testSets = []
	saves = []
	predFNames = []
	#load pairs
	pos = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Guo_Data_Yeast_Tian/guo_yeast_pos_idx.tsv','int')
	neg = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Guo_Data_Yeast_Tian/guo_yeast_neg_idx.tsv','int')
		
	#create folds
	k = kfolds
	trainSets,testSets = PPIPUtils.createKFolds(pos,neg,k,seed=1)

	hyp = {'fullGPU':True}

	for i in range(0,k):
		saves.append(directory+'Guo_Data_Yeast_Tian_'+str(i)+'.out')
		predFNames.append(directory+'Guo_Data_Yeast_Tian_'+str(i)+'_predict.tsv')
		
	return trainSets,testSets, saves,predFNames, currentDir+'PPI_Datasets/Guo_Data_Yeast_Tian/'


def loadPanHumanLarge(directory,kfolds=5):
	allPairs = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Pan_Human_Data/Pan_Large/allPairs.tsv','int')
	trainSets, testSets = PPIPUtils.createKFoldsAllData(allPairs,kfolds)
	saves=[]
	pfs = []
	for i in range(0,kfolds):
		saves.append(directory+'Pan_Human_Large_'+str(i)+'.out')
		pfs.append(directory+'Pan_Human_Large_'+str(i)+'_predictions.tsv')
		
	return trainSets,testSets, saves,pfs, currentDir+'PPI_Datasets/Pan_Human_Data/Pan_Large/'

def loadPanHumanSmall(directory,kfolds=5):
	allPairs = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Pan_Human_Data/Pan_Small/allPairs.tsv','int')
	trainSets, testSets = PPIPUtils.createKFoldsAllData(allPairs,kfolds)
	saves=[]
	pfs = []
	for i in range(0,kfolds):
		saves.append(directory+'Pan_Human_Small_'+str(i)+'.out')
		pfs.append(directory+'Pan_Human_Small_'+str(i)+'_predictions.tsv')
		
	return trainSets,testSets, saves,pfs, currentDir+'PPI_Datasets/Pan_Human_Data/Pan_Small/'
	

def loadPanMartinHuman(directory,kfolds=5):
	allPairs = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Pan_Human_Data/Martin_Human/allPairs.tsv','int')
	trainSets, testSets = PPIPUtils.createKFoldsAllData(allPairs,kfolds)
	saves=[]
	pfs = []
	for i in range(0,kfolds):
		saves.append(directory+'Martin_Human_'+str(i)+'.out')
		pfs.append(directory+'Martin_Human_'+str(i)+'_predictions.tsv')
		
	return trainSets,testSets, saves,pfs, currentDir+'PPI_Datasets/Pan_Human_Data/Martin_Human/'
	
def loadGuoYeastDataChen(directory,kfolds=5):
	trainSets = []
	testSets = []
	saves = []
	predFNames = []
		
	#create folds
	k = kfolds
	
	allPairs = PPIPUtils.parseTSVLst(currentDir+'PPI_Datasets/Guo_Data_Yeast_Chen/protein.actions.tsv',['string','string','int'])
		
	trainSets, testSets = PPIPUtils.createKFoldsAllData(allPairs,kfolds)
	
	for i in range(0,k):
		saves.append(directory+'Guo_Data_Yeast_Chen_'+str(i)+'.out')
		predFNames.append(directory+'Guo_Data_Yeast_Chen_'+str(i)+'_predict.tsv')
		
	return trainSets,testSets, saves,predFNames, currentDir+'PPI_Datasets/Guo_Data_Yeast_Chen/'


def loadLiADData(directory):
	trainSets = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Li_AD/li_AD_train_idx.tsv','int')
	testSets = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Li_AD/li_AD_test_idx.tsv','int')
	trainSets = [np.asarray(trainSets)]
	testSets = [np.asarray(testSets)]
	saves = [directory+'Li2020_AD.out']
	predFNames = [directory+'Li2020_AD_predict.tsv']
	
	return trainSets, testSets, saves,predFNames, currentDir+'PPI_Datasets/Li_AD/'
	
		
		
def loadRichouxHumanDataStrict(directory):
	trainSets = PPIPUtils.parseTSVLst(currentDir+'PPI_Datasets/Richoux_Human_Data/train_pairs_strict.tsv',['string','string','int'])
	testSets = PPIPUtils.parseTSVLst(currentDir+'PPI_Datasets/Richoux_Human_Data/test_pairs_strict.tsv',['string','string','int'])
	trainSets = [np.asarray(trainSets)]
	testSets = [np.asarray(testSets)]
	saves = [directory+'Richoux_Human_Strict.out']
	predFNames = [directory+'Richoux_Human_Strict_predict.tsv']
	return trainSets, testSets, saves,predFNames, currentDir+'PPI_Datasets/Richoux_Human_Data/'
	
		
def loadGuoMultiSpeciesChen(directory,ident='All',kfolds=5):
	if ident in ['01','10','25','40']:
		allPairs = PPIPUtils.parseTSVLst(currentDir+'PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.filtered.'+ident+'.tsv',['string','string','int'])
	else:
		allPairs = PPIPUtils.parseTSVLst(currentDir+'PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.tsv',['string','string','int'])
	
	#create folds
	k = kfolds
	trainSets, testSets = PPIPUtils.createKFoldsAllData(allPairs,kfolds)
	
	saves = []
	predFNames = []
	for i in range(0,k):
		saves.append(directory+'Guo_MultSpecies_Chen'+str(i)+'.out')
		predFNames.append(directory+'Guo_MultSpecies_Chen'+str(i)+'_predict.tsv')
	
	return trainSets, testSets, saves,predFNames, currentDir+'PPI_Datasets/Guo_MultiSpecies_Chen/'
	
def loadLiuFruitFly(directory,kfolds=5):
	trainSets = []
	testSets = []
	saves = []
	predFNames = []
		
	#create folds
	k = kfolds
	
	allPairs = PPIPUtils.parseTSVLst(currentDir+'PPI_Datasets/Liu_Fruit_Fly/pairLst.tsv',['string','string','int'])
		
	trainSets, testSets = PPIPUtils.createKFoldsAllData(allPairs,kfolds)
	
	for i in range(0,k):
		saves.append(directory+'Liu_Fruit_Fly_'+str(i)+'.out')
		predFNames.append(directory+'Liu_Fruit_Fly_'+str(i)+'_predict.tsv')
		
	return trainSets,testSets, saves,predFNames, currentDir+'PPI_Datasets/Liu_Fruit_Fly/'

	
		
def loadDuYeast(directory,kfolds=5,balanced=True):
	allPairs = PPIPUtils.parseTSVLst(currentDir+'PPI_Datasets/Du_Yeast/pairLst.tsv',['string','string','int'])
	
	#create folds
	trainSets, testSets = PPIPUtils.createKFoldsAllData(allPairs,kfolds,balanced=balanced)
	
	saves = []
	predFNames = []
	for i in range(0,kfolds):
		saves.append(directory+'Du_Yeast'+str(i)+'.out')
		predFNames.append(directory+'Du_Yeast'+str(i)+'_predict.tsv')
	
	return trainSets, testSets, saves,predFNames, currentDir+'PPI_Datasets/Du_Yeast/'

def loadJiaYeast(directory,trainDataPerClass=5943,full=True,seed=1,kfolds=5):
	
	trainSets = []
	testSets = []
	saves = []
	predFNames = []
	
	data = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Jia_Data_Yeast/allPairs.tsv')
	fullData = np.asarray(data)
	
	#grab and shuffle all positive and negative data
	dataClasses = np.asarray(fullData[:,2],dtype=np.int32)
	posIdx = np.where(dataClasses==1)[0]
	negIdx = np.where(dataClasses==0)[0]
	np.random.seed(seed)
	np.random.shuffle(posIdx)
	np.random.shuffle(negIdx)
	
	if trainDataPerClass=='Max':
		trainDataPerClass = min(posIdx.shape[0],negIdx.shape[0])
	#split n (trainDataPerClass) points from each class as the crossfold/training data
	trainData = fullData[np.hstack((posIdx[:trainDataPerClass],negIdx[:trainDataPerClass]))]
	testData = fullData[np.hstack((posIdx[trainDataPerClass:],negIdx[trainDataPerClass:]))]
	
	#gen cross fold data
	trainSets,testSets = PPIPUtils.createKFoldsAllData(trainData.tolist(),kfolds)
	
	if full:
		#gen full train/test data
		trainSets.append(trainData)
		testSets.append(testData)
	
	for i in range(0,kfolds+1):
		saves.append(directory+'Jia_Yeast'+str(i)+'.out')
		predFNames.append(directory+'Jia_Yeast'+str(i)+'_predict.tsv')
		
	return trainSets, testSets, saves, predFNames, currentDir+'PPI_Datasets/Jia_Data_Yeast'

	
def getPartial(data,percentage=0.1,seed=5):
	posData = np.where(data[:,2]==1)[0]
	negData = np.where(data[:,2]==0)[0]
	np.random.seed(5)
	np.random.shuffle(posData)
	np.random.shuffle(negData)
	posAmt = int(posData.shape[0]*percentage)
	negAmt = int(negData.shape[0]*percentage)
	fullIdx = np.hstack((posData[:posAmt],negData[:negAmt]))
	newData = data[fullIdx,:]
	return newData


	
def augmentAll(trainSets,testSets):
	retSets = []
	for s in [trainSets,testSets]:
		if s is None:
			retSets.append(None)
		else:
			newSets = []
			for i in range(0,len(s)):
				newSets.append(augment(s[i]))
			retSets.append(newSets)
	return retSets
		
#create pair Y,X for every pair X,Y, and return only unique pairs
def augment(data):
	curData = np.asarray(data)
	#create mirror duplicate
	#index with None to replace dimension loss when indexing, and transpose to restore normal orientation
	curData2 = np.hstack((curData[None,:,1].T,curData[None,:,0].T,curData[None,:,2].T))
	#stack original and duplicate
	curData = np.vstack((curData,curData2))
		
	#remove duplicates, in case of (x,x) pairs, or pairs (x,y) and (y,x) being in the original data
	curData = np.unique(curData,axis=0)
	return curData
		

def convertToFolder(lst):
	lst2 = []
	for item in lst:
		item = '.'.join(item.split('.')[:-1]) #remove suffix
		lst2.append(item + '/')
	return lst2
		
	
	
	
