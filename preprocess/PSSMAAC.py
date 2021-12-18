import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import torch

from PreprocessUtils import loadPSSM

def PSSMAAC(fastas, directory, processPSSM=True,deviceType='cpu'):
	deviceType='cpu'#no speed up to move data to gpu for a single summation
	retVals = []
	header = ['protein']
	for i in range(0,20):
		header.append('a'+str(i))
	retVals.append(header)


	#calculate the sum of all PSSM data
	for item in fastas:
		name = item[0]
		seq = item[1]
		data = loadPSSM(name, seq, directory,usePSIBlast=processPSSM)
		AACVals = (torch.tensor(data).sum(dim=0)/len(seq)).tolist()
		retVals.append([name]+AACVals)

	return retVals