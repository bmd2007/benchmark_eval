#Amino Acid Composition, set to take N groups (default is each of the 20 regular amino acid in its own group) and a group len (default of 1)
#will count the number of times each unique group of length group length occurs
#if sorting is True, if using group length of 3, then sequences AGV, AVG, VAG, GAV, GVA, and VGA will all group to the same bin
#if flip is True and sorting is False, then sequences AGV and VGA will match, sequences VAG and GAV will match, and sequences GVA and AVG will match, but all six will not group to the same bin
#if exclude same, when group length % 2 == 0, if the grouping found is the same group repeated (such as 2,2 or  3,6,3,6), it will not be counted

import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import torch
from PreprocessUtils import getAALookup
import math

def AACounter(fastas, groupings = None, groupLen=1, sorting=False,flip=False,normType='100',separate=False,excludeSame=False,deviceType='cpu'):
	AA =getAALookup()
	
	if groupings is not None:
		AA = groupings
	
	allGroups = set()
	for k,v in AA.items():
		allGroups.add(v)
		
	#return data
	retData = []
	if separate:
		rawData = []
	
	if groupLen%2 != 0 : #can't exclude groups with repeate values on odd length
		excludeSame= False
		
	#add header
	header = ['protein']
	groupMapping = {}
	for i in range(0,len(allGroups)**groupLen):
		idxs = []
		for j in range(0,groupLen):
			idxs.append((i//(len(allGroups)**j))%len(allGroups))
		idxs = idxs[::-1]
		if sorting:
			if sorted(idxs) != idxs:
				continue
		elif flip:
			if idxs > idxs[::-1]:
				continue
		if excludeSame:
			if idxs[len(idxs)//2:] == idxs[:len(idxs)//2]:
				continue
		header.append('g_'+'_'.join(str(k) for k in idxs))
		groupMapping[tuple(idxs)] = len(groupMapping)
		
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
		#get the amino idx for each letter in the seqeunce, with 20 for missing data
		stIdx = torch.tensor([AA.get(st[k],-1) for k in range(0,len(st))]).to(deviceType)
		#remove missing data
		stIdx = stIdx[stIdx!=-1]
		#create list for return values, starting with protein name
		fastaVals = [name]
		
		#convert to matrix, remove letters from end if groupLen>1
		tData= stIdx[:stIdx.shape[0]-(groupLen-1)].unsqueeze(0)
		for i in range(1,groupLen):
			newDat = stIdx[i:stIdx.shape[0]-(groupLen-i-1)].unsqueeze(0)
			tData = torch.cat((tData,newDat),dim=0)
			
		#flip matrix, so that each row contains groupLen consective group indices
		tData = tData.T	
		
		#if sorting, sort each row
		if sorting:
			tData = tData.sort(dim=1)[0]
		#if flipping, for each row, find if a < flip(a)
		elif flip:
			#i don't know any good way to detect, per row, in torch, if a <= flip(a), for a given matrix
			#will think about later
			pass
		
		#get unique groupings and counts
		unique, counts = tData.unique(dim=0,return_counts=True)
		unique = unique.tolist()
		counts = counts.tolist()
		
		#create bins for each count item
		binCounts = torch.zeros(len(groupMapping),device=deviceType)
		
		for itemIdx in range(0,len(unique)):
			item = unique[itemIdx]
			c = counts[itemIdx]
			#couldn't do it in torch, will check it here
			if flip:
				if item > item[::-1]:
					item = item[::-1]
			#find groupIdx for item, add number of occurances
			item = tuple(item)
			
			#if exclude same, skips groups made of the same values
			if excludeSame:
				if item[len(item)//2:] == item[:len(item)//2]:
					continue
			
			#if this crashes, we have an error, since all combinations should be mapped
			binCounts[groupMapping[item]] += c
			
	
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
	