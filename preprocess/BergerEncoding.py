#Code based on python DScript library (pip install dscript )
#The primary difference is saving each protein to a file rather than saving one large h5py file
#With 20,000 sequences, the total data size can balloon over 200gb, which is hard to load from a single file
#MIT License

#Copyright (c) 2020 Samuel Sledzieski

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter
from PreprocessUtils import getPretrainedBergerModel
import numpy as np
#add parent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import PPIPUtils
import torch
from joblib import dump, load

def BergerEncoding(fastas, folder, encodeMap = None, sleepTime=0, deviceType='cpu'):
	#standard encoding used by Berger and Berger From 'Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.
	#as used by DScript
	if encodeMap is None:
		encodeMap = {}
		for item in 'ARNDCQEGHILKMFPSTWYVX':
			encodeMap[item] = len(encodeMap)
		encodeMap['O'] = 11
		encodeMap['U'] = 4
		encodeMap['B'] = 20
		encodeMap['Z'] = 20
	
	AA = encodeMap
	
	PPIPUtils.makeDir(folder)
	
	#get the model used to create the final encoding vectors
	model = getPretrainedBergerModel()
	torch.nn.init.normal_(model.proj.weight)
	model.proj.bias = torch.nn.Parameter(torch.zeros(100))
	model = model.to(deviceType)
	model.eval()
    
	idx = 0
	#run model
	with torch.no_grad():
		for item in fastas:
			#protein name
			name = item[0]
			#sequence
			st = item[1]
			
			if os.path.exists(folder+name):
				continue #already have computation
			

			#get the amino idx for each letter in the seqeunce, with -1 for missing data (indices should be positive. . .)
			stIdx = torch.tensor([AA.get(st[k],-1) for k in range(0,len(st))]).to(deviceType)
			#remove missing data/letters
			stIdx = stIdx[stIdx!=-1]
			
			#run model
			stIdx = stIdx.long().unsqueeze(0)
			stIdx = model.transform(stIdx)
			stIdx = stIdx.squeeze(0)
			stIdx = stIdx.cpu().numpy()
			dump(stIdx,folder+name+'.joblibdump')
			
			idx += 1
			if idx % 10 == 0:
				time.sleep(sleepTime)
			
	return None
