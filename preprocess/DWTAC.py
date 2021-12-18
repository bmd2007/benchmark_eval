

import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from Covariance import Covariance
import pywt
import numpy as np

#for default AAIndex IDs, using close approximatation of Guo's calculations

#Guo AAC calculation
#aaIndexParameters = []
#aaIndexParameters.append('GUO_H1') #hydrophobicity, Added Guo's values to AAIndex.  Cannot match these to values in AAIndex.
#aaIndexParameters.append('HOPT810101') #hydrophicility  (Guo listed 1 value as 0.2, but in aaindex it is 2.  Assuming Guo mistyped, but this could affect papers that copied values from their supplementary data)
#aaIndexParameters.append('KRIW790103') #volumes of side chains of amino acids
#aaIndexParameters.append('GRAR740102') #polarity
#aaIndexParameters.append('CHAM820101') #polarizability
#aaIndexParameters.append('ROSG850103_GUO_SASA') #solvent-accessible surface area (SASA)  Added Guo's Values to AAIndex, taken from Rose et.al 1985, from same table that provides ROSG850101 and ROSG850101, but Guo uses standard state surface area instead
#aaIndexParameters.append('GUO_NCI') #net charge index (NCI) of side chains of amino acid	Cannot find paper these came from


def DWTAC(fastas, waveletType='db1',levels=4,aaIDs = ['GUO_H1','HOPT810101','KRIW790103','GRAR740102','CHAM820101','ROSG850103_GUO_SASA','GUO_NCI'], deviceType='cpu'):

	oneHotVals = []
	for item in aaIDs:
		oneHotVals.append(Covariance(fastas,[item],separate=True,calcType='Lookup',deviceType=deviceType)[1][1:])
		
	header = []
	for item in aaIDs:
		for i in range(0,levels+1):
			for item2 in ['max','mean','min','std']:
				header.append(item+'_'+str(i)+'_'+item2)
	retVals = [header]
	for i in range(0,len(fastas)):
		curVal = [fastas[i][0]] #name of protein
		for j in range(0,len(oneHotVals)):
			#print(oneHotVals[j][i][0].numpy())
			dwt = pywt.wavedec(oneHotVals[j][i][0].numpy(),waveletType,level=levels)
			for k in range(0,len(dwt)): #len dwt should equals levels + 1
				curVal.append(np.max(dwt[k]))
				curVal.append(np.mean(dwt[k]))
				curVal.append(np.min(dwt[k]))
				curVal.append(np.std(dwt[k]))
		retVals.append(curVal)
	return retVals