import os
import sys
#add parent
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import PPIPUtils
import random
import copy

#Functions to generate randomly selected pairs, held out datasets, split proteins into groups, and any other functions needed to create our datasets from a list of proteins and interactions.


#generate random pairs of proteins from a full list of proteins, skipping those in an excluded set
#note, each pair in exclusion set should be sorted as (smallID, largeID)
def genRandomPairs(proteinLst,numPairs,exclusionLst = []):
	exclusionSet = set(exclusionLst)
	retLst = set()
	while len(retLst) < numPairs:
		x = tuple(sorted(random.sample(proteinLst,2)))
		if x not in exclusionSet and x not in retLst and x[0] != x[1]:
			retLst.add(x)
	return list(retLst)

#generate random pairs of proteins with one protein from each of two lists, skipping those in an excluded set
def genRandomPairsAB(proteinLstA,proteinLstB,numPairs,exclusionLst = []):
	exclusionSet = set(exclusionLst)
	retLst = set()
	while len(retLst) < numPairs:
		a = random.sample(proteinLstA,1)[0]
		b = random.sample(proteinLstB,1)[0]
		x = tuple(sorted([a,b]))
		if x not in exclusionSet and x not in retLst and x[0] != x[1]:
			retLst.add(x)
	return list(retLst)

#split a set of proteins into groups randomly, and return a list of the groups, and dictionary mapping proteins to group IDs
def createProteinGroups(proteinLst,numGroups):
	proteinLst = list(set(proteinLst)) #make copy before shuffling to not alter original
	random.shuffle(proteinLst)
	retLst = []
	for i in range(0,numGroups):
		start = (len(proteinLst)* i)//numGroups
		end = (len(proteinLst)* (i+1))//numGroups
		if i == numGroups-1:
			end = len(proteinLst)
		retLst.append(proteinLst[start:end])
	retDict = {}
	for i in range(0,len(retLst)):
		for item in retLst[i]:
			retDict[item] = i
	return retLst, retDict
	
#given a list of pairs, as well as a list and dictionary of groups, creates a dictionary of dictionary of the N groups, and mapped protein pairs to the appropriate double indexed dictionary
def assignPairsToGroups(pairLst,groupsLst,groupDict):
	pairGroups = []
	for i in range(0,len(groupsLst)):
		pairGroups.append([])
		for j in range(0,len(groupsLst)):
			pairGroups[-1].append(set())
	
	for item in pairLst:
		idx1 = groupDict.get(item[0],-1)
		idx2 = groupDict.get(item[1],-1)
		if idx1 == -1:
			print('Error, cannot find item ',item[0])
			exit(42)
		if idx2 == -1:
			print('Error, cannot find item ',item[1])
			exit(42)
		m = min(idx1,idx2)
		m2 = max(idx1,idx2)
		pairGroups[m][m2].add(item)
	return pairGroups

#given a list of pairs, a number of pairs to draw, and a list of pairs to exclude, draws num pairs without repeating, and returns those as a new list
#assumes numPairs < len(pairLst)
def drawPairs(pairLst,numPairs,skipLst = []):
	if len(skipLst) > 0:
		pairLst = list(set(pairLst)-set(skipLst))
	else:
		pairLst = list(set(pairLst))
	random.shuffle(pairLst)
	return pairLst[:numPairs]
	
#takes a file name, list of positive data, and list of negative data, and write the data to a file order randomly or sorted order (default is random)
def writePosNegData(fname,pos,neg,randomOrder=True):
	lst = []
	for item in pos:
		lst.append((item[0],item[1],1))
	for item in neg:
		lst.append((item[0],item[1],0))
	if randomOrder:
		random.shuffle(lst)
	else:
		lst.sort()
	PPIPUtils.writeTSV2DLst(fname,lst)




#generate numPos and numNeg positive and negative pairs randomly, create k folds of data, (optionally) create a folder=foldername, and save the folds to files in the folder
#folderName should end in '/' or '\\'
#intLst and protLst are known interactions (tuple pairs, where for each pair (X,Y), X < Y), and protLst is the list of all proteins
def createRandomKFoldData(intLst, protLst,numPos,numNeg,k,folderName):
	PPIPUtils.makeDir(folderName)
	pos = drawPairs(intLst,numPos)
	neg = genRandomPairs(protLst,numNeg,intLst)
	trainSets, testSets = PPIPUtils.createKFolds(pos,neg,k,seed=None)
	for i in range(0,k):
		PPIPUtils.writeTSV2DLst(folderName+'Train_'+str(i)+'.tsv',trainSets[i])
		PPIPUtils.writeTSV2DLst(folderName+'Test_'+str(i)+'.tsv',testSets[i])

#Primary function for creating non-heldout train and test data
#generats random splits of non-overlapping train and test data, and writes the groups to a file in the given folder
#function generates 1 folder, X train set, and X*N test sets where n is the length of the 2nd argument of tuple pairs
#intLst and protLst are known interactions (tuple pairs, where for each pair (X,Y), X < Y), and protLst is the list of all proteins
#ratioslst a tuple contiaing the following information in the format (A,B,F),[(C1,D1,F1)...]):
#lst tuple arg1, tuple (A,B,F)
#A -- number of positive pairs in training data, integer
#B -- number of negative pairs in training data, integer
#F -- file prefix for training data
#lst tuple arg2, lst of tuples [(C1,D1,E1),(C2,D2,E2),(C3,D3,E3). . .]
#C1,2,3... number of positive pairs in test data set (1,2,3...)
#D1,2,3... number of negative pairs in test data set (1,2,3...)
#E1,2,3... file prefix for test data set (1,2,3...)
#numSets is the number of times to iterate the ratiosLst (generating numSets train, and numSets *len(arg2) test sets)
#folderName is the name of the folder to save the data to
def createRandomData(intLst, protLst, ratiosLst,numSets,folderName):
	PPIPUtils.makeDir(folderName)
	for k in range(0,numSets):
		#generate train data
		trainPosPairs = ratiosLst[0][0]
		trainNegPairs = ratiosLst[0][1]
		trainFNamePrefix = ratiosLst[0][2]
		trainPos = drawPairs(intLst,trainPosPairs)
		trainNeg = genRandomPairs(protLst,trainNegPairs,intLst)
		writePosNegData(folderName+trainFNamePrefix+str(k)+'.tsv',trainPos, trainNeg)
		
		#generate the test data:
		for item in ratiosLst[1]:
			testPosPairs = item[0]
			testNegPairs = item[1]
			testFNamePrefix = item[2]
			testPos = drawPairs(intLst,testPosPairs,trainPos)
			testNeg = genRandomPairs(protLst,testNegPairs,intLst+trainNeg)
			writePosNegData(folderName+testFNamePrefix+str(k)+'.tsv',testPos, testNeg)
			
			
			


#Primary function for spliting proteins in groups
#splits proteins into numGroups groups, and creates a grid of which interactions fall into which groups
#continues randomly generating groups until the minimum size requirements are met
#minSizeSmall is the minimum size for group pairs where i=j, such as Group (0,0)
#minSizeLarge is the minimum size for group pairs where i!=j, such as Group (0,1)
#if you set the minimum sizes too large for the given number of groups, this functon will fail and return None, None, None
#since this will generate numGroups small groups, and (numGroups*numGroups-numGroups)/2 large groups,
#we recommend not setting minsizeSmall >= len(intLst)/((numGroups*numGroups-numGroups)/2+numGroups/2)/2 * .97, and not setting minSizeLarge more than twice that
#For example, given 120,000 interactions, and 6 groups, points will be split into 15 large groups and 6 small/half-size groups
#Given 120,000 interactions, and that 120,000/18/2*.97  = 3233.  Keeping the numbers below 3233 (small) and 6466 (large) is recommended, but lower numbers could be needed to converge

#returns list of list of proteins per group, dictionary mapping proteins to groups, and list of list of group pairs with sets of interactions
def createGroupData(intLst,protLst,numGroups,minSizeSmall,minSizeLarge,maxAttempts=1000):
	groups = []
	groupDict = {}
	groupInts = []
	for attempt in range(0,maxAttempts):
		#create groups, returning list of groups, and dictionary mapping proteins to groups
		groups, groupDict = createProteinGroups(protLst,numGroups)
		smallest1 = minSizeSmall
		smallest2 = minSizeLarge
		#split all interaction pairs into group pairs, returning list of list of groups
		groupInts = assignPairsToGroups(intLst,groups,groupDict)
		for i in range(0,len(groupInts)):
			for j in range(i,len(groupInts[i])):
				if i == j:
					smallest1 = min(smallest1,len(groupInts[i][j]))
				else:
					smallest2 = min(smallest2,len(groupInts[i][j]))	
		#ensure smallest groups meet size requirements
		if smallest1 >= minSizeSmall and smallest2 >=minSizeLarge:
			return groups, groupDict, groupInts
	else:
		return None, None, None



#Primary function for creating heldout train and test data
#generates held out sizes of non-overlapping train and test data, such that no protein in the training data appears in the test data,
#and writes the sets to files.  The number of sets will be of size (len(groups)*len(groups)+len(groups))/2
#function generates 1 folder, X train set, and X*N test sets where n is the length of the 2nd argument of tuple pairs
#groups and groupInts are lists containing groups of proteins, and the known interactions per groupPair
#ratioslst is a tuple contianing the following information (A,B,C,D,E,F),[(A1,D1,E1,F1)...]
#lst tuple arg1, tuple (A,B,C,D,E,F)
#A -- number of positive pairs per small group when held out group is small, 'all' for all
#B -- number of positive pairs per large group when held out group is small, 'all' for all
#C -- number of positive pairs per small group when held out group is large, 'all' for all
#D -- number of positive pairs per large group when held out group is large, 'all' for all
#E -- multiplier of number of positive pairs to calculate number of negative pairs
#F -- file prefix for training data
#lst tuple arg2, lst of tuples [(A1,D1,E1,F1),(A2,D2,E2,F2)...]
#arguments are same as training data tuple
def genCrossHeldOutDataWithGroups(groups,groupInts,ratiosLst,folderName):
	PPIPUtils.makeDir(folderName)
	#loop through grid, holding out data (i,j)
	for i in range(0,len(groups)):
		for j in range(i,len(groups)):
			trainPos = []
			trainNeg = []
			testPos = []
			testNeg = []
			for item in ratiosLst[1]:
				testPos.append([])
				testNeg.append([])
			#loop through grid (index k,l) given held out coordinates (i,j)
			for k in range(0,6):
				for l in range(k,6):
					#if this is the subgroup we are holding out, get the test data
					if (k,l) == (i,j):
						testIdx = -1
						for item in ratiosLst[1]:
							testIdx+=1
							#get test data
							#50/50
							if k==l:
								numPairs = item[0]
							else:
								numPairs = item[1]
								
							if numPairs == 'all':
								testPos[testIdx] += list(groupInts[k][l])
							else:
								testPos[testIdx] += drawPairs(groupInts[k][l],numPairs)
							testNeg[testIdx] += genRandomPairsAB(groups[k],groups[l],int(len(testPos[testIdx])*item[2]),groupInts[k][l])
						continue
					#if this groups contains either of the held out sets (but not both, as that would be the heldout subgroup (k,l)==(i,j)), skip it
					elif k in [i,j] or l in [i,j]:
						continue
					#data not being held out for testing, create train set
					#get train data
					#calculate number of positive interactions to take from each section
					
					#small Held out
					if i == j:
						#small set
						if k == l:
							ratePos = ratiosLst[0][0]
						#large set
						else:
							ratePos = ratiosLst[0][1]
					#large held out
					else:
						#small set
						if k == l:
							ratePos = ratiosLst[0][2]
						#large set
						else:
							ratePos = ratiosLst[0][3]
					
					if ratePos == 'all':
						pos = groupInts[k][l]
					else:
						pos = drawPairs(groupInts[k][l],ratePos)
					
					#generate negative pairs, holding out positive interactions
					neg = genRandomPairsAB(groups[k],groups[l],int(len(pos)*ratiosLst[0][4]),groupInts[k][l])
					trainPos += pos
					trainNeg += neg
					
			writePosNegData(folderName+ratiosLst[0][5]+str(i)+'_'+str(j)+'.tsv',trainPos,trainNeg)
			print('train',i,j,len(trainPos),len(trainNeg))
			testIdx = -1
			for item in ratiosLst[1]:
				testIdx += 1
				writePosNegData(folderName+item[3]+str(i)+'_'+str(j)+'.tsv',testPos[testIdx],testNeg[testIdx])
				print('test',i,j,testIdx,len(testPos[testIdx]),len(testNeg[testIdx]))
			