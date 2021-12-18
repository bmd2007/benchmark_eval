import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from CTD_Composition import CTD_Composition
from CTD_Distribution import CTD_Distribution
from CTD_Transition import CTD_Transition
from PreprocessUtils import STDecode
from PreprocessUtils import MCDEncode

#default groups are based on conjoint triad method
def MCDCTD(fastas, numSplits=4, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'], deviceType='cpu'):


	encoded, encodedSize = MCDEncode(fastas,numSplits)
		
	comp = CTD_Composition(encoded,groupings,deviceType)
	tran = CTD_Transition(encoded,groupings,deviceType)
	dist = CTD_Distribution(encoded,groupings,deviceType)
	
	comp = STDecode(comp,encodedSize)
	tran = STDecode(tran,encodedSize)
	dist = STDecode(dist,encodedSize)
		
	return (comp, tran, dist)
	
	
	
	
	

