import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter


def ConjointTriad(fastas, deviceType='cpu'):
	groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C']
	groupLen=3
	groupMap = {}
	idx = 0
	for item in groupings:
		for let in item:
			groupMap[let] = idx
		idx += 1
	return AACounter(fastas,groupMap,groupLen,normType='CTD',deviceType=deviceType)
