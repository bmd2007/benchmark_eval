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

def genAllFasta():
	
	#file lst, containing train, val, test data in strict set, as well as 
	fLst = [currentDir+'double-medium_1166_train_mirror.txt',currentDir+'double-medium_1166_val_mirror.txt',currentDir+'test_double_mirror.txt',currentDir+'medium_1166_train_mirror.txt',currentDir+'medium_1166_val_mirror.txt',currentDir+'medium_1166_test_mirror.txt']

	pairLst = []
	proteinData = {}
	for fname in fLst:
		f = open(fname)
		header = None
		pairLst.append([])
		for line in f:
			#skip header line
			if header is None:
				header = line
				continue
			line = line.strip().split()
			protA = line[0]
			protB = line[1]
			seqA = line[2]
			seqB = line[3]
			classData = line[4]
			
			
			for pair in ((protA,seqA),(protB,seqB)):
				if pair[0] not in proteinData:
					proteinData[pair[0]] = pair[1]
			pairLst[-1].append((protA,protB,classData))

		f.close()

	#write unique sequences
	f = open(currentDir+'allSeqs.fasta','w')
	finalProtSet = set()
	for item in proteinData:
		if len(proteinData[item])<=30:
			continue #skip small sequences
		finalProtSet.add(item)
		f.write('>'+item+'\n'+proteinData[item]+'\n')
	f.close()


	def writePairsToFile(lst,fname):
		f=  open(fname,'w')
		for item in lst:
			if item[0] in finalProtSet and item[1] in finalProtSet:
				f.write('\t'.join(item)+'\n')
		f.close()
			

	#write training data for strict set
	writePairsToFile(pairLst[0]+pairLst[1],currentDir+'train_pairs_strict.tsv')

	#write test data for strict set
	writePairsToFile(pairLst[2],currentDir+'test_pairs_strict.tsv')

	#write training data for regular set
	writePairsToFile(pairLst[3]+pairLst[4],currentDir+'train_pairs_regular.tsv')

	#write test data for regular set
	writePairsToFile(pairLst[5],currentDir+'test_pairs_regular.tsv')



if __name__ == '__main__':
	genAllFasta()