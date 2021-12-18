import gzip
import shutil
import argparse
import os
import os.path
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

import PPIPUtils

def createPPISet(geneHist=parentDir+'ProteinSetCreation/gene_history_2021',biogridPPIs=currentDir+'BioGridOrganisms/BIOGRID_PARSED_Human_4.3.194.tsv',dataGenesFile=parentDir+'DataGeneIDs2021.tsv',outputFilePPI=currentDir+'/DataPPIs2021.tsv'):
	if not os.path.isfile(geneHist):
		PPIPUtils.downloadFile('https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_history.gz', geneHist+'.gz')
		with gzip.open(geneHist+'.gz', 'rb') as f_in:
			with open(geneHist, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)

	f = open(geneHist)
	disCont = {}
	h = None
	for line in f:
		if not h:
			h = line
			continue
		line = line.strip().split()
		newGeneID = int(line[1]) if line[1] != '-' else '-'
		oldGeneID = int(line[2])
		disCont[oldGeneID] = newGeneID
	f.close()


	f = open(dataGenesFile)
	proteinSet = set()
	h = None
	for line in f:
		if not h:
			h = line
			continue
		line = line.strip().split()
		if line[0] in disCont:
			print('dicont uni',line[0])
			line[0] = disCont[line[0]]
		proteinSet.add(int(line[0]))
	f.close()

	count = [0,0,0,0,0,0,0]
	f = open(biogridPPIs)
	h = None
	intSet = set()
	for line in f:
		if not h:
			h = line
			continue
		line = line.strip().split()
		try:
			id1 = int(line[3])
			id2 = int(line[4])
			tax1 = int(line[7])
			tax2 = int(line[8])
			evi = line[10]
			if tax1 != 9606 or tax2 != 9606:
				count[0] += 1
				continue
			if id1 == id2:
				count[1] += 1
				continue
			if evi == 'Protein-RNA':
				count[2] += 1
				continue
			if id1 not in proteinSet or id2 not in proteinSet:
	#			print('Warning, cannot find protein from interaction in uniprot data')
	#			print(line)
				count[3] += 1
				continue  #for now, lets remove genes we can't map to uniprot.  Some are long non-protein encoding RNA, others are psuedo/predicted genes (unsure how they validated an interaction)
				#proteinSet.add(id1)
				#proteinSet.add(id2)



			idCop = (min(id1,id2),max(id1,id2))
			if id1 in disCont:
				if disCont[id1] == '-':
					count[4] += 1
					continue #obsolete protein
				else:
					id1 = disCont[id1]
					
			if id2 in disCont:
				if disCont[id2] == '-':
					count[4] += 1
					continue #obsolete protein
				else:
					id2 = disCont[id2]
			idNew = (min(id1,id2),max(id1,id2))
			if idCop != idNew:
				print(idCop,idNew)


			
			pair = (min(id1,id2),max(id1,id2))

			if pair in intSet:
				count[5] += 1
				continue
				
			intSet.add(pair)
			count[6] += 1
		except:
			continue
	f.close()
	print('Removed due to non-human protein',count[0])
	print('Removed due to self-interaction',count[1])
	print('Removed due to Protein-RNA binding',count[2])
	print('Removed due to non-uniprot protein',count[3])
	print('Removed due to obsolete protein ID',count[4])
	print('Removed due to duplicate interaction',count[5])
	print('Remaining Protein Interactions',count[6])

	f2 = open(outputFilePPI,'w')
	for item in intSet:
		f2.write('\t'.join(str(s) for s in item) + '\n')
	f2.close()




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Creates list of PPIs using BioGrid Parsed data (mapped through gene history), and list of all human genes (combination of genes in PPIs with uniprot protein genes)')
	parser.add_argument('--GeneHistory',help='Gene History file from NCBI',default=parentDir+'ProteinSetCreation/gene_history_2021')
	parser.add_argument('--BiogridPPIs',help='Parsed list of Biogrid PPIs',default=currentDir+'BioGridOrganisms/BIOGRID_PARSED_Human_4.3.194.tsv')
	parser.add_argument('--DataGenes',help='File containing list of genes mapping to protein in uniprot',default=parentDir+'DataGeneIDs2021.tsv')
	parser.add_argument('--OutputFilePPI',help='File to write List of PPIs to.',default=currentDir+'/DataPPIs2021.tsv')
	args = parser.parse_args()
	createPPISet(args.GeneHistory,args.BiogridPPIs,args.DataGenes,args.OutputFilePPI)