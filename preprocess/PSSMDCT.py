import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import numpy as np
from scipy.fft import dctn

from PreprocessUtils import loadPSSM


def PSSMDCT(fastas, directory, dctType = 2, dctNormType = 'ortho', keepVals=20,deviceType='cpu',processPSSM=False):
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
		data = np.asarray(data)
		d = dctn(data,type=dctType,norm=dctNormType)
		
		#grab top keepVals X 20 values, by default 400 = 8000
		d = d[:keepVals,:].reshape(keepVals*d.shape[1])
		d = d.tolist()
		retVals.append([name]+d)
	return retVals