import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter


def AAC(fastas, groupings = None, groupLen=1, normType='100',deviceType='cpu'):
	if groupings is not None:
		groupMap = {}
		idx = 0
		for item in groupings:
			for let in item:
				groupMap[let] = idx
			idx += 1
	else:
		groupMap = None
		
	return AACounter(fastas,groupMap,groupLen,normType=normType,deviceType=deviceType)
