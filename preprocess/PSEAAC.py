import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter
from Covariance import Covariance
from PairwiseDist import PairwiseDist
import torch
def PSEAAC(fastas, aaIDs=['GUO_H1','HOPT810101','CHOU_SIDE_MASS'],pairwiseAAIDs=None,pairwiseScale='Abs1',calcType='AvgSq',lag=10,w=0.05,amphipathic=False,deviceType='cpu'):

	#PSEAAC basically has two components
	#Component 1, Amino Acid Count, length 20, normalized
	#Component 2, Average of the Average Squared difference of amino acid properties, usually using a lag of 10
	#if calculating amphipathic PSEAAC (APSEAAC), then the values aren't averaged, so component 2 will be of size lag*len(aaIDs) instead of just lag
	#Component 2 is multiplied by a weight, w
	#all 30 values are then divided so that their sum is equal to 1
	
	#component 1, count all length 1 sequences in the fasta files, normalize data to add to 1 (100%), return names and counts seperated

	(names,counts) =  AACounter(fastas,separate=True,deviceType=deviceType)
	
	#component 2, get avg of squared values across given lags for each aaID, return names and values seperated
	if aaIDs is not None and pairwiseAAIDs is None:
		(names,values) = Covariance(fastas,aaIDs,lag,separate=True,calcType='AvgSq',deviceType=deviceType)
	elif pairwiseAAIDs is not None and aaIDs is None:
		(names,values) =  PairwiseDist(fastas,pairwiseAAIDs,lag,separate=True,calcType='AvgSq',scaleType=pairwiseScale,deviceType=deviceType)
		aaIDs = pairwiseAAIDs
	else: #need to get values for both
		(names,values) = Covariance(fastas,aaIDs,lag,separate=True,calcType='AvgSq',deviceType=deviceType)
		(names2,values2) =  PairwiseDist(fastas,pairwiseAAIDs,lag,separate=True,calcType='AvgSq',scaleType=pairwiseScale,deviceType=deviceType)
		for i in range(0,len(values)):
			values[i].extend(values2[i])
		aaIDs = aaIDs + pairwiseAAIDs
			
			
	countsHeader = counts[0]
	valheader = values[0]
	
	values = torch.tensor(values[1:]).to(deviceType)
	counts = torch.tensor(counts[1:]).to(deviceType)
	#if amphipathic, keep value of each autcovariance function, creating vector of size 20+L*lag
	#else, average across all autcovariance functions, creating vector of size 20+lag
	#if not amphipathic, get average across all aaIDs
	if not amphipathic:
		newVals = torch.zeros((values.shape[0],lag),device=deviceType)
		for i in range(0,len(aaIDs)):
			newVals += values[:,(i*lag):((i+1)*lag)]
		newVals = newVals/len(aaIDs)
	else:
		newVals = values
	
	#multiply component 2 by w
	newVals = newVals * w
	
	#concat the values into a single value set, of size len(fastas), (20+lag)
	finalVals = torch.cat((counts,newVals),dim=1)
	
	#normalize
	finalVals = finalVals/(finalVals.sum(axis=1,keepdims=True))
	
	#combine finalVals with names, and return
	finalVals = [countsHeader+valheader] + finalVals.tolist()
	retData = []
	for i in range(0,len(names)):
		retData.append(names[i] + finalVals[i])
	return retData
	
	
	
	
	
	

