#Amino Acid Composition, set to take N groups (default is each of the 20 regular amino acid in its own group) and a group len (default of 1)
#will count the number of times each unique group of length group length occurs
#if sorting is True, if using group length of 3, then sequences AGV, AVG, VAG, GAV, GVA, and VGA will all group to the same bin
#if flip is True and sorting is False, then sequences AGV and VGA will match, sequences VAG and GAV will match, and sequences GVA and AVG will match, but all six will not group to the same bin
#if exclude same, when group length % 2 == 0, if the grouping found is the same group repeated (such as 2,2 or  3,6,3,6), it will not be counted.  
#if getRawValues is true, return a list containing 1 tensor per string, where the tensor values map to the group's idx.  Also returns group mapping
#gap and truncate are used if grabbing only gapped indices of data.  For example, using non-overlaping windows on a group length of 3 by grabbing only every gap=3 rd group, 
#truncate determines where to remove values from to ensure sequence length is a multiple of gap.
#masking allows for gapped matches, such as 10011101, or 11111101, but all masks must be the same length for the current code
#maskWeights should be None of have the same length as masking.  Tese weights are multipled by the final counts from each mask.
#If using masking, if flattenRawValues is True, then values will be flattened before returning (with unneeded values removed).  If false, unfiltered matrices will be returned.
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import torch
from PreprocessUtils import getAALookup
import math

def calcI(lst,numGroups):
	x = lst[0]
	for i in range(1,len(lst)):
		x = x * numGroups + lst[i]
	return x

def AACounter(fastas, groupings = None, groupLen=1, sorting=False,flip=False,normType='100',separate=False,excludeSame=False,getRawValues=False,flattenRawValues=True,gap=0,truncate='right',masking = None, maskWeights = None, deviceType='cpu'):
	#set device type = cpu for these calculations, as they occur protein by protein and take too long to transfer to gpu
	#in future work, may look at ways to run multiple proteins at once, by:
	#extending length of all proteins to be equal, with extended values mapping to the find counting bucket (which gets tossed)
	#and make a row of buckets per sequence (which means all bucketIdxs would have to add len(buckets)*protein_num to their value
	#for now, this seems to complicated to be needed
	deviceType = 'cpu'
	AA =getAALookup()
	
	if groupings is not None:
		AA = groupings
	
	allGroups = set()
	for k,v in AA.items():
		allGroups.add(v)
		
	numGroups = len(allGroups)
	
	#return data
	retData = []
	if separate:
		rawData = []
	
	if groupLen%2 != 0 : #can't exclude groups with repeate values on odd length
		excludeSame= False


	maskingMatrix = None
	if masking: #all masks must be same length.  May modify this later if necessary, but it is simplier to code this way.
		if maskWeights is not None and len(maskWeights) != len(masking):
			print('Error, mask weights must be the same length as masking')
			exit(42)
			
		maskLen = len(masking[0])
		for item in masking:
			if len(item) != maskLen:
				print('Error, all masks must be the same length')
				exit(42)
			if item.count('1') != groupLen:
				print('Error, non-group length mask found')
				exit(42)
		maskingMatrix = []
		
		for i in range(0,maskLen):
			maskingMatrix.append([])
		
		for i in range(0,len(masking)):
			for j in range(0,maskLen):
				if masking[i][j] == '1':
					maskingMatrix[j].append(i)
		for i in range(0,maskLen):
			maskingMatrix[i] = torch.tensor(maskingMatrix[i]).to(deviceType)
	else:
		maskWeights = None

	#check gap parameters
	if gap > 0:
		truncate = truncate.lower()
		if truncate not in ['right','left','midright','midleft']:
			print('error, unrecognized gap type')
			exit(42)
				

	#add header
	header = ['protein']
	#number of possible combinations
	comboNum = numGroups**groupLen
	
	groupMapping = {}
	groupLookup = []
	for i in range(0,comboNum):
		idxs = []
		for j in range(0,groupLen):
			idxs.append((i//(numGroups**j))%numGroups)
		idxs = idxs[::-1]
		if sorting:
			if sorted(idxs) != idxs:
				groupLookup.append(groupMapping[calcI(sorted(idxs),numGroups)])
				continue
		elif flip:
			if idxs > idxs[::-1]:
				groupLookup.append(groupMapping[calcI(idxs[::-1],numGroups)])
				continue
		if excludeSame:
			if idxs[len(idxs)//2:] == idxs[:len(idxs)//2]:
				groupLookup.append(-1)
				continue
		header.append('g_'+'_'.join(str(k) for k in idxs))
		groupLookup.append(len(groupMapping))
		groupMapping[calcI(idxs,numGroups)] = len(groupMapping)
		
	if separate:
		retData.append([header[0]])
		rawData.append(header[1:])
	else:
		retData.append(header)
	
	groupLookup = torch.tensor(groupLookup)
	groupLookup[groupLookup==-1] = len(groupMapping)
	
	idx = -1
	for item in fastas:
		idx += 1
		#protein name
		name = item[0]
		#sequence
		st = item[1]
		#get the amino idx for each letter in the seqeunce, with -1 for missing data
		stIdx = torch.tensor([AA.get(st[k],-1) for k in range(0,len(st))]).to(deviceType).long()
		#remove missing data
		stIdx = stIdx[stIdx!=-1]
		#create list for return values, starting with protein name
		fastaVals = [name]

		
		if masking is not None:
			tData = torch.zeros(stIdx.shape[0]-(maskLen-1),device=deviceType).long()
			#calculate indices from stIdx, using same logic as calcI, using matrix for gaps
			tDataMat = torch.zeros((len(masking),tData.shape[0])).long()
			for i in range(0,maskLen):
				tDataMat[maskingMatrix[i]] = tDataMat[maskingMatrix[i]] * numGroups + stIdx[i:(stIdx.shape[0]-(maskLen-i-1))]
			tData = tDataMat.T #ensure amino acid sequence length is in dimension 0, to match gapping code when no matrix is supplied
		else:
			tData = torch.zeros(stIdx.shape[0]-(groupLen-1),device=deviceType).long()
			#calculate indices from stIdx, using same logic as calcI
			for i in range(0,groupLen):
				tData = tData * numGroups + stIdx[i:(stIdx.shape[0]-(groupLen-i-1))]
			
		#map from allIdx to actual index (which takes into account sorting/flipping/excluding) using group lookup array
		tData2 = groupLookup[tData.flatten()].reshape(tData.shape)
		
		#if gap > 0, only keep every nth group
		if gap > 0:
			#gap should divide into tData2.shape[0] -1
			extra = (tData2.shape[0]-1) - (((tData2.shape[0]-1)//gap)*gap)
			#if gap doesn't divide evenly in sequence length, check where to remove values from
			if extra > 0:
				extra1 = 0
				extra2 = 0
				if truncate == 'right':
					extra2 = extra
				elif truncate == 'left':
					extra1 = extra
				elif truncate == 'midleft':
					extra2 == extra//2
					extra1 = extra-extra2
				elif truncate == 'midright':
					extra1 = extra//2
					extra2 = extra-extra1
				tData2 = tData2[extra1:(tData2.shape[0]-extra2)]
			tData2 = tData2[torch.arange(0,tData2.shape[0],gap)]
				
		
		if getRawValues: #if get Raw Values, just return the indices mapping each letter to a group, to be used by some other function
			#remove invalid indices
			if masking is None or flattenRawValues:
				tData2 = tData2.flatten()
				tData2 = tData2[tData2!=len(groupMapping)]
			#append data to return list
			fastaVals.append(tData2)
			retData.append(fastaVals)
			#skip rest of loop
			continue
		
		
		#create bins for each count item
		binCounts = torch.zeros(len(groupMapping)+1,device=deviceType)

		if maskWeights:
			for i in range(0,len(maskWeights)):
				#add all values to bins, per mask
				binCounts2 = torch.zeros(binCounts.shape,device=deviceType)
				z = tData2[:,i]
				binCounts2.index_add_(0,z,torch.ones(z.shape,device=deviceType))
				binCounts += binCounts2 * maskWeights[i]
		else:
			tData2 = tData2.flatten() #flatten from matrix format, if masking is used but weights are not
			#add all values to bins
			binCounts.index_add_(0,tData2,torch.ones(tData2.shape[0],device=deviceType))
		
		#remove last value, which is used for excluded groups and is unneeded
		binCounts = binCounts[:-1]
		
		if normType == '100':
			#normalize so all numbers add to 1
			binCounts = binCounts/binCounts.sum()
		elif normType == 'CTD':			
			#normalize, based on shen's paper
			m1 = binCounts.min()
			m2 = binCounts.max()
			binCounts = (binCounts-m1)/m2
		elif normType == 'SeqLen': #equalivalent to 100 in most cases if excludeSame=False
			divisor = stIdx.shape[0]-(groupLen-1)
			binCounts = binCounts/divisor
		elif normType is None:
			pass #do nothing
		
		binCounts = binCounts.tolist()
		
		if not separate:
			fastaVals += binCounts
			retData.append(fastaVals)
		else:
			rawData.append(binCounts)
			retData.append(fastaVals)
	

	if separate:
		retData = (retData,rawData)
		
	return retData
	