import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from PSEAAC import PSEAAC

def QuasiSequenceOrder(fastas, pairwiseAAIDs=['Grantham','Schneider-Wrede'],pairwiseScale='Abs1',lag=10,w=0.1,deviceType='cpu'):
	#just call PSEAAC, they are basically the same
	return PSEAAC(fastas, aaIDs=None,pairwiseAAIDs=pairwiseAAIDs,pairwiseScale=pairwiseScale,lag=lag,w=w,amphipathic=False,deviceType=deviceType)
	
	
	
	
	
	
	

