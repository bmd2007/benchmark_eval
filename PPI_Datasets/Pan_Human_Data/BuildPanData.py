import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
currentDir = currentdir+'/'
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentDir = parentdir+'/'
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)

import PPIPUtils



def parseFile(fname,headerLines, allPairs, allSeqs, idMap):
	f = open(fname)
	curData = []
	for line in f:
		if headerLines > 0:
			headerLines -= 1
			continue
		line = line.strip()
		#each protein pair is 5 lines
		#idx/identifiers
		#id 1
		#sequence
		#id 2
		#sequence
		if len(curData) < 5:
			curData.append(line)
		if len(curData) == 5:
			#parse
			curIdxs = []
			if len(curData[2]) <= 30 or len(curData[4]) <= 30:
				curData = []
				continue
			for i in [1,3]:
				curData[i] = curData[i][1:] #remove >
				if curData[i] in idMap: #if alreadyParsed
					curIdxs.append(idMap[curData[i]])
				else:
					#add new entry
					idMap[curData[i]] = len(idMap)
					curIdxs.append(idMap[curData[i]])
					#add sequence
					allSeqs[curIdxs[-1]] = curData[i+1]
				#print(idMap)
				#print(curData)
			allPairs.append(tuple(curIdxs))
			curData = []
	f.close()
	return
		
def writeIDMap(fname,idMapping):
	f = open(fname,'w')
	f.write('ID\tProtein')
	lst = []
	for k,v in idMapping.items():
		lst.append((v,k))
	lst.sort()
	for item in lst:
		f.write('\n'+str(item[0])+'\t'+item[1])
	f.close()			

def writeSeqs(fname,seqs):
	f=open(fname,'w')
	keys = sorted(seqs.keys())
	for item in seqs:
		f.write('>'+str(item)+'\n')
		f.write(seqs[item]+'\n')
	f.close()

def writePairs(fname,pairsPos,pairsNeg):
	f = open(fname,'w')
	for item in pairsPos:
		f.write(str(item[0])+'\t'+str(item[1])+'\t1\n')
	for item in pairsNeg:
		f.write(str(item[0])+'\t'+str(item[1])+'\t0\n')
	f.close()



def genAllFasta():
	#Large set
	allPairsPos = []
	allSeqs = {}
	idMap = {}
	parseFile(currentDir+'Supp-A.txt',4,allPairsPos,allSeqs,idMap)
	allPairsNeg = []
	parseFile(currentDir+'Supp-B.txt',4,allPairsNeg,allSeqs,idMap)
	PPIPUtils.makeDir(currentDir+'Pan_Large/')
	writeIDMap(currentDir+'Pan_Large/IDMapping.tsv',idMap)
	writeSeqs(currentDir+'Pan_Large/allSeqs.fasta',allSeqs)
	writePairs(currentDir+'Pan_Large/allPairs.tsv',allPairsPos,allPairsNeg)


	#Small Set
	allPairsPos = []
	allSeqs = {}
	idMap = {}
	parseFile(currentDir+'Supp-C.txt',4,allPairsPos,allSeqs,idMap)
	allPairsNeg = []
	parseFile(currentDir+'Supp-D.txt',3,allPairsNeg,allSeqs,idMap)
	PPIPUtils.makeDir(currentDir+'Pan_Small/')
	writeIDMap(currentDir+'Pan_Small/IDMapping.tsv',idMap)
	writeSeqs(currentDir+'Pan_Small/allSeqs.fasta',allSeqs)
	writePairs(currentDir+'Pan_Small/allPairs.tsv',allPairsPos,allPairsNeg)


	#Martin Human
	allPairs = []
	allSeqs = {}
	idMap = {}
	parseFile(currentDir+'Supp-E.txt',5,allPairs,allSeqs,idMap)
	PPIPUtils.makeDir(currentDir+'Martin_Human/')
	writeIDMap(currentDir+'Martin_Human/IDMapping.tsv',idMap)
	writeSeqs(currentDir+'Martin_Human/allSeqs.fasta',allSeqs)
	writePairs(currentDir+'Martin_Human/allPairs.tsv',allPairs[:len(allPairs)//2],allPairs[len(allPairs)//2:])


if __name__ == '__main__':
	genAllFasta()