import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import torch


def Random(fastas, randomLen=500, minRand=0,maxRand=1,deviceType='cpu'):
	retVals = []
	retVals.append(['protein']+[str(i) for i in range(0,500)])
	for item in fastas:
		(name,st) = item
		rand = (torch.rand((randomLen))*(maxRand-minRand))+minRand
		lst = [name] + rand.cpu().tolist()
		retVals.append(lst)
	
	return retVals
