import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import torch
from PreprocessUtils import loadAAData
import numpy as np

def Covariance(fastas, aaIDs, lag=30, separate=False, calcType='AutoCovariance', deviceType='cpu'):

	#set device type = 'cpu' for these calculations, since they go protein by protein
	#in the future, we made run multiple proteins at once on the gpu, but, we woulc have
	#to find a way to pad the data and ensure that the for loops don't run into the padding data
	#which seems overly complicated for now
	deviceType = 'cpu'
	
	AA, AANames, AAData = loadAAData(aaIDs)
	
	#normalize all features
	for AAIdx in range(0,len(AAData)):
		featVals = torch.tensor(AAData[AAIdx]).to(deviceType)
		#normalize mean
		featVals = featVals - featVals.mean()
		#divide by std  (std = square of mean norm, averaged, then apply sqrt)
		featVals = featVals / (featVals.pow(2).mean().pow(.5))
		AAData[AAIdx] = featVals
	
	
	#calculate the sum of squares for all features, needed for some types of AC
	sumSqs = []
	for AAIdx in range(0,len(AAData)):
		sumSqs.append(AAData[AAIdx].pow(2).sum())
		
	#return data
	retData = []
	#if seperate, keep row headers and data in seperate lists
	if separate:
		rawData = []
		
	#add header
	header = ['protein']
	for item in AANames:
		for j in range(1,lag+1):
			header.append('d'+item+'_'+str(j))
			
	if separate:
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
			#get the normalize feature values
			featVals = AAData[featIdx]
			#get the vals for each letter using the stIdx list
			stVals = featVals[stIdx]
			
			#if we just want the value for each amino acid, with no further computations
			if calcType == 'Lookup':
				fastaVals.append(stVals)
				continue
				
			if calcType =='Moran' or calcType == 'Geary' or calcType == 'AutoCovariance':
				stMean = stVals.mean()
				stVals2 = stVals -stMean
				squaredVal = stVals2.pow(2).mean()
			
			#for all lags, calc the average of all but the first lag vals * all but the last lag values
			for j in range(1,lag+1):
				if calcType == 'AutoCovariance':
					fastaVals.append((stVals2[:-j]*stVals2[j:]).mean().item())
				elif calcType == 'NMBroto': #same as autocovariance, but on an unnormalized sequence
					fastaVals.append((stVals[:-j]*stVals[j:]).mean().item())
				elif calcType == 'AvgSq': #used for PSeaaC
					fastaVals.append((stVals[:-j]-stVals[j:]).pow(2).mean().item())
				elif calcType == 'Moran': #same as autocovariance, but divide by squared of sequence values
					v1 = (stVals2[:-j]*stVals2[j:]).mean()
					fastaVals.append((v1/squaredVal).item())
				elif calcType == 'Geary': #same as AvgSq, but divided by 2 * normalized sequence squared mean
					v1 = (stVals[:-j]-stVals[j:]).pow(2).mean()
					fastaVals.append((v1/(2*squaredVal)).item())
					
			#exit(42)
				
		if not separate:
			retData.append(fastaVals)
		else:
			rawData.append(fastaVals[1:])
			retData.append([fastaVals[0]])
		
	if separate:
		retData = (retData,rawData)
		
	return retData
	