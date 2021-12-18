import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import torch
from PreprocessUtils import loadPairwiseAAData


#Used to calculate Covariance like formulas using pairwise distance matrices instead of per amino acid values
#By default, we are scaling the feature matrices to have an maximum absolute value of 1 (ABS1)
#This ensures that different features have about the same values, and matches well with the Schnedier-Wrede matrix, initially used by Chou

#Also defaulting to AvgSq aggregation, for similar reason
def PairwiseDist(fastas, pairwiseAAIDs, lag=30, separate=False, calcType='AvgSq', scaleType='Abs1', deviceType='cpu'):

	#set device type = 'cpu' for these calculations, since they go protein by protein
	#in the future, we made run multiple proteins at once on the gpu, but, we woulc have
	#to find a way to pad the data and ensure that the for loops don't run into the padding data
	#which seems overly complicated for now
	deviceType = 'cpu'
	
	AA, AANames, AAData = loadPairwiseAAData(pairwiseAAIDs)
	for AAIdx in range(0,len(AAData)):
		featVals = torch.tensor(AAData[AAIdx]).to(deviceType)
		if scaleType is not None:
			if scaleType == 'Abs1':
				featVals = featVals/featVals.abs().max(axis=1,keepdims=True)[0] #[0] to get values, [1] to get indices
			elif scaleType == 'Abs1Sq':
				featVals = featVals/featVals.abs().max() #max over all rows
			elif scaleType == 'Norm':
				#normalize mean
				featVals = featVals - featVals.mean(axis=1,keepdims=True)
				#divide by std  (std = square of mean norm, averaged, then apply sqrt)
				featVals = featVals / (featVals.pow(2).mean().pow(.5))
			elif scaleType == 'NormSq': #across all rows
				#normalize mean
				featVals = featVals - featVals.mean()
				#divide by std  (std = square of mean norm, averaged, then apply sqrt)
				featVals = featVals / (featVals.pow(2).mean().pow(.5))
		AAData[AAIdx] = featVals
	
		
	#return data
	retData = []
	#if seperate, keep row headers and data in seperate lists
	if separate or scaleType=='All':
		rawData = []
		
	#add header
	header = ['protein']
	for item in AANames:
		for j in range(1,lag+1):
			header.append('d'+item+'_'+str(j))
			
	if separate or scaleType=='All':
		retData.append([header[0]])
		rawData.append(header[1:])
	else:
		retData.append(header)

	for item in fastas:
		#protein name
		name = item[0]
		#sequence
		st = item[1]
		#get the amino idx for each letter in the seqeunce, with -1 for missing data (indices should be positive. . .)
		stIdx = torch.tensor([AA.get(st[k],-1) for k in range(0,len(st))]).to(deviceType)
		#remove missing data
		stIdx = stIdx[stIdx!=-1]
		#create list for return values, starting with protein name
		fastaVals = [name]
		#for every aaID
		for featIdx in range(0,len(AANames)):
			#get the feature data
			featVals = AAData[featIdx]
			
			
			#for all lags, run calculations
			for j in range(1,lag+1):
				idx0 = stIdx[:-j]
				idx1 = stIdx[j:]
				vals = featVals[idx0,idx1]
				
				if calcType == 'SumSq':
					fastaVals.append((vals**2).sum().item())
				elif calcType == 'AvgSq':
					fastaVals.append((vals**2).mean().item())
				
				
		if not separate and scaleType != 'All':
			retData.append(fastaVals)
		else:
			rawData.append(fastaVals[1:])
			retData.append([fastaVals[0]])
		
	if separate:
		retData = (retData,rawData)
		
	return retData
	