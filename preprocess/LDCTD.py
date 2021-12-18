import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from CTD_Composition import CTD_Composition
from CTD_Distribution import CTD_Distribution
from CTD_Transition import CTD_Transition
from PreprocessUtils import LDEncode10
from PreprocessUtils import STDecode

#default groups are based on conjoint triad method
def LDCTD(fastas, encodeFunc=LDEncode10, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'], deviceType='cpu'):


	if encodeFunc is not None:
		encoded, encodedSize = encodeFunc(fastas)
		
	comp = CTD_Composition(encoded,groupings,deviceType)
	tran = CTD_Transition(encoded,groupings,deviceType)
	dist = CTD_Distribution(encoded,groupings,deviceType)
	
	if encodeFunc is not None:
		comp = STDecode(comp,encodedSize)
		tran = STDecode(tran,encodedSize)
		dist = STDecode(dist,encodedSize)
		
	return (comp, tran, dist)
	
	
	
	
	

