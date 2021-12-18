#based on paper by Zhang, Wang, and Wang, "A New Encoding Scheme to Improve the Performance of Protein Structural Class Prediction" 2005
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter
import torch

from PreprocessUtils import stringSplitEncodeEGBW
from PreprocessUtils import STDecode
from PreprocessUtils import getAALookup


#Default groupings are based on 3 possible arrangements of 4 groups from paper (shown below) containing two sets in each group
#neuter and non-polarity residue C1={G,A,V,L,I,M,P,F,W}
#neuter and polarity residue C2={Q,N,S,T,Y,C}
#acidic residue C3={D,E}
#alkalescence residue C4={H,K,R}
def EGBW(fastas, lSize=11, groupLsts = [['GAVLIMPFWQNSTYC','DEHKR'],['GAVLIMPFWDE','QNSTYCHKR'],['GAVLIMPFWHKR','QNSTYCDE']], deviceType='cpu'):

	#create lookup of letters, and convert all letters to indices
	
	#all letters in groupLsts, indexed into their own groups
	letterLookup = {}
	#group lsts, converted to their index values
	groupLstsIdxs = []
	#header for returnVals
	header = []
	for i in range(0,len(groupLsts)):
		header.append(str(i))
		for j in range(0,len(groupLsts[i])):
			newLst2 = []
			for letter in groupLsts[i][j]:
				if letter not in letterLookup:
					letterLookup[letter] = len(letterLookup)
				newLst2.append(letterLookup[letter])
			#only keep ones data (first group per groups lst)
			if j == 0:
				groupLstsIdxs.append(newLst2)

	

	#encode strings given ldSize
	encoded = stringSplitEncodeEGBW(fastas,lSize)
		
	#since the values of this function are just the frequency each group of letters appears (versus the total length), just calculate the frequency of each letter once
	(names, vals) = AACounter(encoded, groupings=letterLookup, normType=None,separate=True,deviceType=deviceType)
	
	
	#throw away headers, and convert vals to torch
	vals = torch.tensor(vals[1:]).to(deviceType)
	
	#create tensor to contain actual return values
	#-1 for header
	rawVals = torch.zeros((len(names)-1,len(header)),device=deviceType)
	
	#string lengths
	lengths = vals.sum(axis=1,keepdims=True)

	#get values of frequency of a group of letters appears
	#equal to sum of appearance of all leters divided by length of string
	idx = 0
	for i in range(0,len(groupLstsIdxs)):
		rawVals[:,idx] = (torch.sum(vals[:,groupLstsIdxs[i]],axis=1,keepdims=True)/lengths).squeeze(1)
		idx+=1
		
	rawVals = rawVals.tolist()
			
			
	#combine finalVals with names
	finalVals = [header] + rawVals
	retData = []
	for i in range(0,len(names)):
		retData.append(names[i] + finalVals[i])
	
	#decode the data
	retData = STDecode(retData,lSize)
	
	#return
	return retData

