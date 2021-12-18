import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
currentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)

from AACounter import AACounter

#by default, use CTD groupings
#Based on paper Prediction of protein folding class using global description of amino acid sequence  INNA DUBCHAK, ILYA MUCHNIKt, STEPHEN R. HOLBROOK, AND SUNG-HOU KIM
#Based on paper Prediction of protein allergenicity using local description of amino acid sequence, by Joo Chuan Tong, Martti T. Tammi
#First Paper listed multiplied calculed percentaged by 100 before using them, but I'm leaving them as floats between 0-1 as that matches most other data types better.
def CTD_Distribution(fastas, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'], deviceType='cpu'):
	if groupings is not None:
		groupMap = {}
		idx = 0
		for item in groupings:
			for let in item:
				groupMap[let] = idx
			idx += 1
	else:
		groupMap = None
		
	#i don't know a good way to do this in torch/numpy, or I would try to
	#I could convert each sequence to torch, and then call torch.where for each group, but that doesn't feel any faster than pure python
	#since I do the conversions from letter to sequence in python anyway
	
	retVals = []
	header = ['protein']
	
	#hold indices each group occurs at in sequence
	idxCount = {}
	for k,v in groupMap.items():
		idxCount[v] = []
		
	groups = sorted(idxCount.keys())
	
	for group in groups:
		for item in ['_1','_25','_50','_75','_100']:
			header.append(str(group)+item)
	retVals.append(header)
	for item in fastas:
		#protein name
		name = item[0]
		#sequence
		st = item[1]
		#create list for return values, starting with protein name
		fastaVals = [name]
		
		#reset idx count
		for group in idxCount:
			idxCount[group] = []
			
		#idx for current letter, don't increment if letter not in groupMap
		letterIdx = 0
		for let in st:
			groupIdx = groupMap.get(let,-1)
			if groupIdx == -1:
				continue
			idxCount[groupIdx].append(letterIdx)
			letterIdx += 1
			
		
		#for each possible group, find where 1st, 25%, 50%, 75%, and last index of that group occurs
		#and divide by sequence length (letterIdx) (same as normType='100')
		for group in groups:
			vals = idxCount[group]
			if len(vals) == 0:
				vals = [0] #just make all values 0 if it didn't appear
			totalCount = len(idxCount[group])
			valIdxs = [0,round((len(vals)-1)*.25),round((len(vals)-1)*.5),round((len(vals)-1)*.75),-1]
			for item in valIdxs:
				fastaVals.append(vals[item]/(letterIdx-1))
		retVals.append(fastaVals)
		
	return retVals