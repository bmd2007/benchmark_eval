import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import torch
from joblib import dump, load
from PreprocessUtils import loadPSSM
import numpy as np

#creates a single file, containing a list of all PSSMs
def PSSMLST(fastas, directory, saveFolder, processPSSM=True, saveFile='PSSMLst.pssms',deviceType='cpu'):
	deviceType = 'cpu' #not enough speedup to move data per fasta item for simple multiplication
	retVals = []


	nameLst = []
	pssmLst = []

	#calculate the sum of all PSSM data
	for item in fastas:
		name = item[0]
		seq = item[1]
		data = loadPSSM(name, seq, directory, usePSIBlast=processPSSM)
		nameLst.append(name)
		pssmLst.append(np.asarray(data))


	tup = (nameLst,pssmLst)
	dump(tup,saveFolder+saveFile)