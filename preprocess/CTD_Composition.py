import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter

#by default, use CTD groupings
#Based on paper Prediction of protein folding class using global description of amino acid sequence  INNA DUBCHAK, ILYA MUCHNIKt, STEPHEN R. HOLBROOK, AND SUNG-HOU KIM
#Based on paper Prediction of protein allergenicity using local description of amino acid sequence, by Joo Chuan Tong, Martti T. Tammi,
#First Paper listed multiplied calculed percentaged by 100 before using them, but I'm leaving them as floats between 0-1 as that matches most other data types better.
def CTD_Composition(fastas, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'],deviceType='cpu'):
	if groupings is not None:
		groupMap = {}
		idx = 0
		for item in groupings:
			for let in item:
				groupMap[let] = idx
			idx += 1
	else:
		groupMap = None
		
	return AACounter(fastas,groupMap,normType='100',deviceType=deviceType)
