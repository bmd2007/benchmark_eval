import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import torch


def Chaos(fastas, gridShape=(4,4)):
	retVals = []
	header = ['protein']
	for i in range(0,gridShape[0]):
		for j in range(0,gridShape[1]):
			header.append('a'+str(i)+'_a'+str(j))
	retVals.append(header)
	aaLookup = {'A':'GCT','G':'GGT','M':'ATG','S':'TCA','C':'TGC','H':'CAC','N':'AAC','T':'ACT','D':'GAC','I':'ATT','P':'CCA','V':'GTG','E':'GAG','K':'AAG','Q':'CAG','W':'TGG','F':'TTC','L':'CTA','R':'CGA','Y':'TAC'}

	coordinates0 = {'T':0,'C':0,'A':gridShape[0],'G':gridShape[0]}
	coordinates1 = {'T':0,'C':gridShape[1],'A':0,'G':gridShape[1]}

	for item in fastas:
		name = item[0]
		seq = item[1]
		seq = ''.join([aaLookup.get(seq[k],'') for k in range(0,len(seq))])
		coord0 = [coordinates0[k] for k in seq]
		coord1 = [coordinates1[k] for k in seq]
		
		grid = torch.zeros(gridShape)
		#start at center
		curPos0 = gridShape[0]/2
		curPos1 = gridShape[1]/2
		
		for i in range(0,len(seq)):
			curPos0 = (curPos0 + coord0[i])/2
			curPos1 = (curPos1 + coord1[i])/2
			grid[min(gridShape[0]-1,int(curPos0)),min(gridShape[1]-1,int(curPos1))] += 1
		grid = grid.flatten()
		grid = grid/grid.sum()
		retVals.append([name]+grid.tolist())
		
	return retVals