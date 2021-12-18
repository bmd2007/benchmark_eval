import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
currentDir = currentdir+'/'

def genAllFasta():
	f = open(currentDir+'Jia_2019.txt')
	headers = [6,3,3]
	mode = 0
	pairs = []
	seqs = {}
	curSeq = ''
	curProt = ''
	for line in f:
		if headers[mode] > 0:
			headers[mode] -= 1
			continue
		if mode in [0,1]:
			line = line.strip()
			if len(line) == 0:
				mode += 1
				continue
			line = line.split()
			for item in line:
				item = item.split('-')
				item.append(str(1-mode)) #positive first, negative second
				pairs.append(tuple(item))
		elif mode == 2:
			if line[0] == '>':
				if len(curSeq)>0:
					if len(curSeq)>30:
						seqs[curProt] = curSeq
				curSeq = ''
				curProt = line[1:].strip()
			else:
				curSeq += line.strip()
	f.close()

	if len(curSeq)>30:
		seqs[curProt] = curSeq

	f = open(currentDir+'allPairs.tsv','w')
	for p in pairs:
		if p[0] in seqs and p[1] in seqs:
			f.write('\t'.join(p)+'\n')
		else:
			print('short',p)
	f.close()


	f = open(currentDir+'allSeqs.fasta','w')
	for k,v in seqs.items():
		f.write('>'+k+'\n'+v+'\n')
	f.close()
			
				
			

if __name__ == '__main__':
	genAllFasta()