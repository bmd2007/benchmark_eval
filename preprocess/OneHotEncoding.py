import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import numpy as np
#add parent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import PPIPUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump, load
from Methods.SimpleDataset import SimpleDataset
from Methods.NetworkRunner import NetworkRunner
from AACounter import AACounter
from torch.utils import data
from joblib import dump, load
import math
import numpy as np
		
		

#1 epoch is more than enough to train a network this small with enough proteins
def OneHotEncoding(fastas, fileName, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'],groupLen=1,sorting=False, flip=False,excludeSame=False,deviceType='cpu'):
	
	if groupings is not None:
		groupMap = {}
		idx = 0
		for item in groupings:
			for let in item:
				groupMap[let] = idx
			idx += 1
		numgroups = len(groupings)
		print(groupMap,numgroups)
	else:
		groupMap = None
		numgroups=20

	parsedData = AACounter(fastas, groupMap, groupLen, sorting=sorting,flip=flip,excludeSame=excludeSame,getRawValues=True,deviceType=deviceType)
	
	#number of unique groups, typically 20 amino acids, times length of our embeddings, typically 1, equals the corpus size
	corpusSize = numgroups*groupLen
	
	f = open(fileName,'w')
	#create Matrix
	for i in range(0,corpusSize):
		lst = [0] * corpusSize
		lst[i] = 1
		f.write(','.join(str(k) for k in lst)+'\n')
	f.write('\n')
	for item in parsedData[1:]:
		name = item[0]
		stVals = item[1].cpu().numpy()
		f.write(name+'\t'+','.join(str(k) for k in stVals) +'\n')
	f.close()
	
	#for item in parsedData[1:]:
	#	name = item[0]
	#	stVals = item[1].cpu().numpy()
	#	data = np.zeros((corpusSize,stVals.shape[0]),dtype=np.int64)
	#	colIdx = np.arange(0,stVals.shape[0],dtype=np.int64)
	#	data[stVals,colIdx] = 1
	#	dump(data,folder+name+'.joblibdump')
	return None

	
