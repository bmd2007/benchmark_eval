import os
import sys
#add parent and grandparent and great-grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)
parentdir3 = os.path.dirname(parentdir2)
sys.path.append(parentdir3)
currentDir = currentdir+'/'
parentDir = parentdir+'/'
import argparse

def createPPIListFromBioGRIDandFasta(seqFileName=parentDir+'allSeqs.fasta',dataPPIFileName=currentDir+'DataPPIs2021.tsv',uniprotEntrezInteractionsFile=currentDir+'HumanUniprotEntrezInteractionLst.tsv'):
	f = open(seqFileName)
	entrezSet = set()
	idx = 0
	for line in f:
		if idx == 0:
			entrezSet.add(line.strip().strip('>'))
		idx += 1
		idx = idx % 2
	f.close()

	f = open(dataPPIFileName)
	f2 = open(uniprotEntrezInteractionsFile,'w')
	for line in f:
		line = line.strip().split('\t')
		if line[0] not in entrezSet or line[1] not in entrezSet:
			print('skipping',line)
		else:
			f2.write(line[0] + '\t' + line[1]+ '\n')
	f.close()
	f2.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generated File List of PPIs combining an Interaction list with a list of valid Genes/Proteins')
	parser.add_argument('--SeqFileName',help='File containing protein sequences',default=parentDir+'allSeqs.fasta')
	parser.add_argument('--DataPPIFileName',help='File containing valid PPis',default=currentDir+'DataPPIs2021.tsv')
	parser.add_argument('--OutputFileName',help='File to write not PPI List to',default=currentDir+'HumanUniprotEntrezInteractionLst.tsv')
	args = parser.parser_args()
	createPPIListFromBioGRIDandFasta(args.SeqFileName,args.DataPPIFileName,args.OutputFileName)