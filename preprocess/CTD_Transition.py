import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from AACounter import AACounter

#by default, use CTD groupings
#Based on paper Prediction of protein folding class using global description of amino acid sequence  INNA DUBCHAK, ILYA MUCHNIKt, STEPHEN R. HOLBROOK, AND SUNG-HOU KIM
#Based on paper Prediction of protein allergenicity using local description of amino acid sequence, by Joo Chuan Tong, Martti T. Tammi,
#First Paper listed multiplied calculed percentaged by 100 before using them, but I'm leaving them as floats between 0-1 as that matches most other data types better.
#Transition should be all transition between group A and B (count(A,B) + count(B,A)), divided by total possible transitions (len(seq)-1)
def CTD_Transition(fastas, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'],deviceType='cpu'):
	if groupings is not None:
		groupMap = {}
		idx = 0
		for item in groupings:
			for let in item:
				groupMap[let] = idx
			idx += 1
	else:
		groupMap = None
		
	#exclude same =True will ensure that a transition from group A to group A will not be counted
	#sorting ensures that Count(A,B) = counts(A,B) + counts(B,A) (either sort or flip being true would work)
	#normType of SeqLen ensures that all values will be divided by len(seq)-(groupLen-1).  This is equalivant to '100' when groupLen=1 and excludeSame=False
	return AACounter(fastas,groupMap,groupLen=2,sorting=True,excludeSame=True,normType='SeqLen',deviceType=deviceType)
