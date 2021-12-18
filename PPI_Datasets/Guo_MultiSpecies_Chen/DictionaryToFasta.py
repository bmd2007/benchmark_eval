import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
currentDir = currentdir+'/'

def genAllFasta():	
	#note, all sequences in dictionary have more than 30 amino acids, the minimum I'm requiring, so I'm doing minimal preprocessing
	f = open(currentDir+'CeleganDrosophilaEcoli.dictionary.tsv')
	f2 = open(currentDir+'allSeqs.fasta','w')
	for line in f:
		line = line.strip().split()
		f2.write('>'+line[0] + '\n' + line[1] +'\n')
	f.close()
	f2.close()

if __name__ == '__main__':
	genAllFasta()