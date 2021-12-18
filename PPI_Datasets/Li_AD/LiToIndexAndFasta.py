import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
currentDir = currentdir+'/'
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
import PPIPUtils
import urllib
import urllib.request
import time


def genAllFasta():
	uniprotFastaLoc, uniprotFastaLoc2 = PPIPUtils.getUniprotFastaLocations()

	trainingData = []
	testingData = []
	allProts = set()
	f = open(currentDir+'ad_train.txt')
	for line in f:
		line = line.strip().split()
		trainingData.append(line)
		for i in range(0,2):
			allProts.add(line[i])
	f.close()
	f = open(currentDir+'ad_test.txt')
	for line in f:
		line = line.strip().split()
		testingData.append(line)
		for i in range(0,2):
			allProts.add(line[i])
	f.close()

	fastas = PPIPUtils.parseUniprotFasta(uniprotFastaLoc, allProts)

	protIdxMapping = {}
	for item in allProts:
		if item not in fastas: 
			#if we can't find the sequence, then we should actually download uniprots unreviewed (TrEMBL) sequences, and combine it with the reviewed set
			#however, that file is 49gbs of data, and we are only missing about 30 seqeunces, so I am just grabbed them fron the site
			#If you hard drive isn't full, you can grab the trembl data instead
			try: 
				url = "https://www.uniprot.org/uniprot/"+item+".fasta"
				lst = urllib.request.urlopen(url).read().decode('utf-8').split('\n')
				fastaSt = ''.join(lst[1:])
				fastas[item] = fastaSt
				time.sleep(1) #always wait after downloading a webpage to not flood the website
			except Exception as ex:#can't find sequence, so skip it
				print(ex)
				print(item,'missing')
				continue
		if len(fastas[item]) <= 30: #skip small sequences
			print(item,fastas[item])
			continue
		protIdxMapping[item] = str(len(protIdxMapping)+1)

	f = open(currentDir+'allSeqs.fasta','w')
	for item in fastas:
		if item in protIdxMapping:
			f.write('>'+protIdxMapping[item]+'\n')
			f.write(fastas[item]+'\n')
	f.close()

	f = open(currentDir+'li_AD_train_idx.tsv','w')
	for item in trainingData:
		idx1 = protIdxMapping.get(item[0],-1)
		idx2 = protIdxMapping.get(item[1],-1)
		c = item[2]
		if idx1 == -1 or idx2 == -1: #can't find a sequence, so skip it
			continue
		f.write(idx1+'\t'+idx2+'\t'+c+'\n')
	f.close()
		
	f = open(currentDir+'li_AD_test_idx.tsv','w')
	for item in testingData:
		idx1 = protIdxMapping.get(item[0],-1)
		idx2 = protIdxMapping.get(item[1],-1)
		c = item[2]
		if idx1 == -1 or idx2 == -1: #can't find a sequence, so skip it
			continue
		f.write(idx1+'\t'+idx2+'\t'+c+'\n')
	f.close()


if __name__ == '__main__':
	genAllFasta()