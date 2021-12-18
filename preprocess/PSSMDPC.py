import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import torch

from PreprocessUtils import loadPSSM


def PSSMDPC(fastas, directory, deviceType='cpu',processPSSM=True):
	deviceType = 'cpu' #not enough speedup to move data per fasta item for simple multiplication
	retVals = []
	header = ['protein']
	for i in range(0,20):
		for j in range(0,20):
			header.append('a'+str(i)+'_a'+str(j))
	retVals.append(header)

	#calculate the sum of all PSSM data
	for item in fastas:
		name = item[0]
		seq = item[1]
		data = loadPSSM(name, seq, directory,usePSIBlast=processPSSM)
		tense = torch.tensor(data).to(deviceType)
		#product of all 20 columns by all 20 columns offset by 1 letter in lag
		results = tense[:-1,:].T @ tense[1:,:]
		#normalize
		results = results/(len(seq)-1)
		
		results = results.reshape(400).tolist()
		
		retVals.append([name]+results)
	return retVals