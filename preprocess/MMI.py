import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter
import torch

from AACounter import AACounter

#default groups are based on conjoint triad method
def MMI(fastas, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'], deviceType='cpu'):

	#MMI is based on the counts of amino acids in given groups of length 1, 2, and 3.  So we will grab those first
	groupMap = {}
	idx = 0
	for item in groupings:
		for let in item:
			groupMap[let] = idx
		idx += 1
	
	names, l1Data = AACounter(fastas,groupMap,1,normType=None,sorting=True,separate=True,deviceType=deviceType)
	l1Headers = l1Data[0]
	l1Data = torch.tensor(l1Data[1:]).to(deviceType)
	names, l2Data = AACounter(fastas,groupMap,2,normType=None,sorting=True,separate=True,deviceType=deviceType)
	l2Headers = l2Data[0]
	l2Data = torch.tensor(l2Data[1:]).to(deviceType)
	names, l3Data = AACounter(fastas,groupMap,3,normType=None,sorting=True,separate=True,deviceType=deviceType)
	l3Headers = l3Data[0]
	l3Data = torch.tensor(l3Data[1:]).to(deviceType)
	
	#we also need the length of all sequences, which we can get from summing the length 1 data
	seqLens = l1Data.sum(dim=1,keepdim=True)
	
	
	#the end goal is to calculate with 7 groups is to get 119 values, of length 1 (7), 2 (28) or 3 (84)
	#make the header
	header = []
	
	#map pair (a,b) to index in l2Data
	dictionaryLookups2 = {}
	
	
	#length 1
	for i in range(0,len(groupings)):
		header.append('g_'+str(i))
	#length 2
	idx = 0
	for i in range(0,len(groupings)):
		for j in range(i,len(groupings)):
			header.append('g_'+str(i)+'_'+str(j))
			dictionaryLookups2[(i,j)] = idx
			dictionaryLookups2[(j,i)] = dictionaryLookups2[(i,j)]
			idx += 1
			
	#length 3
	for i in range(0,len(groupings)):
		for j in range(i,len(groupings)):
			for k in range(j,len(groupings)):
				header.append('g_'+str(i)+'_'+str(j)+'_'+str(k))
	
	rawData = torch.zeros((l1Data.shape[0],len(header)),device=deviceType)
	
	#group length 1 is simply the count /length of their occurances in the sequences
	rawData[:,0:len(groupings)] = l1Data/seqLens
	
	#group length 2 is (occurances+1/(length-1+1)) * log((occurances+1/(length-1+1))/(occurances(a)*occurances(b)))
	#where occurances(a) = (num(a)+1)/(l+1)
	# or p(ab or ba) * log(p(ab or ba)/p(a)/p(b))
	#we use f instead of p to ensure no zeros, where p(a)=count(a)/L, f(a)=(count(a)+1)/(L+1)
	
	
	#add 1, divide by l-1+1, to get frequency calculations
	probLookups = (l1Data+1)/(seqLens+1)
	probLookups2 = (l2Data+1)/(seqLens-1+1)
	probLookups3 = (l3Data+1)/(seqLens-2+1)
	
	idx = len(groupings)
	startIdx = idx
	for i in range(0,len(groupings)):
		for j in range(i,len(groupings)):
			occurances = probLookups2[:,idx-startIdx].unsqueeze(1)
			#get 1 letter probabilities.  .unsqueeze(1) to keep them as vectors instead of arrays
			probI = probLookups[:,i].unsqueeze(1)
			probJ = probLookups[:,j].unsqueeze(1)
			#do log(f(ab or ba)/(f(a)f(b)))
			basicProb = occurances/(probI*probJ)
			logProb = torch.log(basicProb)
			rawData[:,idx] = (occurances * logProb).flatten()
			idx += 1
			
			
			
	#group length 3 is I(a,b) - I(a,b|c), where I(a,b) is what was previously calculated
	#I(a,b|c) = H(a|c) - H(a|b,c)
	#where H(x|y) = -occurances(x,y)/occurances(y) * log(occurances(x,y)/occurances(y))
	startIdx = idx
	for i in range(0,len(groupings)):
		for j in range(i,len(groupings)):
			for k in range(j,len(groupings)):
				#for H(a|c)
				#occurances a,c
				oAC = probLookups2[:,dictionaryLookups2[(i,k)]].unsqueeze(1)
				#occurances c
				oC = probLookups[:,k].unsqueeze(1)
				#for H(a|b,c)
				#occurances (a,b,c)
				oABC = (probLookups3[:,idx-startIdx].unsqueeze(1)+1)/(seqLens-2+1)
				#occurances (b,c)
				oBC = probLookups2[:,dictionaryLookups2[(j,k)]].unsqueeze(1)
				
				#get I(a,b), which is equal to f_ab*log(f_ab/(f_a*f_b))
				iAB = rawData[:,len(groupings)+dictionaryLookups2[(i,j)]].unsqueeze(1)
				
				#get H(a|c), which is equal to -oac/oc*log(oac/oc)
				part_hA_C = oAC/oC
				hA_C = -(part_hA_C)*torch.log(part_hA_C)
				
				#get H(a|b,c), which is equal to -oABC/oBC*log(oABC/oBC)
				part_hA_BC = oABC/oBC
				hA_BC = -(part_hA_BC)*torch.log(part_hA_BC)
				
				#calculate I(a,b|c) = hA_C - hA_BC
				iAB_C = hA_C - hA_BC
				
				#calculate I(a,b,c) = iAB - iAB_C
				iABC = iAB - iAB_C
				
				rawData[:,idx] = iABC.flatten()
				idx += 1

	
	rawData = rawData.tolist()
	
	retData = [[names[0]+header]]
	for i in range(0,len(names)):
		retData.append((names[i]+rawData[i-1]))

	return retData
	
	
	
	
	
	

