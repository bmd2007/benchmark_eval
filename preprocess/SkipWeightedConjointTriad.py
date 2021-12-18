import sys, os
import torch
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter


def SkipWeightedConjointTriad(fastas, masking = ['111','1011','1101'], weights=[1,.5,.5], deviceType='cpu'):
	groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C']
	groupLen=3
	groupMap = {}
	idx = 0
	for item in groupings:
		for let in item:
			groupMap[let] = idx
		idx += 1
	maskMat ={}
	if len(weights) != len(masking):
		print('Error, mismatching lenghts between weights and maskining')
		exit(42)
		
	
	for i in range(0,len(masking)):
		if masking[i].count('1') !=3:
			print('Error, invalid masking for length 3 mapping')
			exit(42)
		if len(masking[i]) not in maskMat:
			maskMat[len(masking[i])] = [[],[]]
		maskMat[len(masking[i])][0].append(masking[i])
		maskMat[len(masking[i])][1].append(weights[i])
	
	
	
	#return AACounter(fastas,groupMap,groupLen,normType='CTD',deviceType=deviceType)
	
	final = None
	names = None
	for maskLen in maskMat:
		masks, weights = maskMat[maskLen]
		#names will be the same every time, its just the string names for the proteins from the fasta file
		#header will be the same each time as well
		names, curVals = AACounter(fastas,groupMap,groupLen,separate=True,masking=masks,maskWeights=weights,normType=None,deviceType=deviceType)
		header = curVals[0]
		curVals = curVals[1:]
		if final is None:
			final = torch.tensor(curVals).to(deviceType)
		else:
			final += torch.tensor(curVals).to(deviceType)
	#run CTD normalization
	#normalize, based on shen's paper
	m1 = final.min(dim=1)[0].unsqueeze(0).T
	m2 = final.max(dim=1)[0].unsqueeze(0).T
	final = (final-m1)/m2
	
	final = [header]+ final.tolist()
	retVals = []
	for i in range(0,len(names)):
		retVals.append(names[i] + final[i])
		
	return retVals