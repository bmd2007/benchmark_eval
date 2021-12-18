import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
currentDir = currentdir+'/'

def genAllFasta():
	#lists containing amino acid sequences for proteins
	lstNA = []
	lstNB = []
	lstPA = []
	lstPB = []
	lsts = [lstNA,lstNB,lstPA,lstPB]
	files = [currentDir+'n_protein_a.txt',currentDir+'n_protein_b.txt',currentDir+'p_protein_a.txt',currentDir+'p_protein_b.txt']

	#open all files, read proteins into lists, keeping order matching
	for fIdx in range(0,4):
		f = open(files[fIdx])
		for line in f:
			line = line.strip()
			if len(line) == 0 or line[0] == '#':
				continue
			lsts[fIdx].append(line)


	#create a dictionary containing each unique protein sequence with a numeric ID associated
	allUniqueProteins = {}
	#create list mapping from idxs to amino acid sequences
	aaLst = []

	#convert amino acid sequences to IDs
	idxNA = []
	idxNB = []
	idxPA = []
	idxPB = []
	idxLsts = [idxNA,idxNB,idxPA,idxPB]

	for lIdx in range(0,4):
		for item in lsts[lIdx]:
			if item not in allUniqueProteins:
				allUniqueProteins[item] = len(allUniqueProteins)
				aaLst.append(item)
			idxLsts[lIdx].append(allUniqueProteins[item])

	#generate fasta file from unique amino acid sequences
	f = open(currentDir+'allSeqs.fasta','w')
	for i in range(0,len(aaLst)):
		f.write('>'+str(i)+'\n')
		f.write(aaLst[i]+'\n')
	f.close()

	#write positive pairs file
	f = open(currentDir+'guo_yeast_pos_idx.tsv','w')
	for i in range(0,len(idxPA)):
		f.write(str(idxPA[i])+ '\t'+str(idxPB[i])+'\n')
	f.close()


	#write negative pairs file
	f = open(currentDir+'guo_yeast_neg_idx.tsv','w')
	for i in range(0,len(idxPA)):
		f.write(str(idxNA[i])+ '\t'+str(idxNB[i])+'\n')
	f.close()


if __name__ == '__main__':
	genAllFasta()