import gzip
import argparse
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


def createEntrezUniprotMapping(uniprotDataFile=currentDir+'HUMAN_9606_idmapping2021.dat',geneInfoFile=currentDir+'Homo_sapiens_2021.gene_info',outputFile=parentDir+'DataGeneIDs2021.tsv'):
	if not os.path.isfile(geneInfoFile):
		PPIPUtils.downloadFile('https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz', geneInfoFile+'.gz')
		with gzip.open(geneInfoFile+'.gz', 'rb') as f_in:
			with open(geneInfoFile, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
				

	if not os.path.isfile(uniprotDataFile):
		PPIPUtils.downloadFile('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz', uniprotDataFile+'.gz')
		with gzip.open(uniprotDataFile+'.gz', 'rb') as f_in:
			with open(uniprotDataFile, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)

	f = open(uniprotDataFile)
	humanGenes = {}
	for line in f:
		line = line.strip().split()	
		if len(line) <= 2:
			continue
		if line[1] == 'GeneID':
			if line[0] not in humanGenes:
				humanGenes[line[0]] = set()
			humanGenes[line[0]].add(line[2])
	f.close()

	f = open(geneInfoFile)
	nameMap = {}
	for line in f:
		line = line.strip().split('\t')
		geneID = line[1]
		geneSym = line[2]
		name = line[8]
		nameMap[geneID] = [geneSym,name]
	f.close()

	#Note, this code only left outer joins uniprot-entrez IDs to entrez Gene Names
	#in future, may want to use inner join, to filter out recently removed Entrez IDs
	f = open(outputFile,'w')
	f.write('Entrez\tSymbol\tName\tEntrez\tUniprot\n')
	for uni in humanGenes:
		for entrez in humanGenes[uni]:
			sym,name = nameMap.get(entrez,['',''])
			f.write('\t'.join([entrez,sym,name,entrez,uni])+'\n')
	f.close()
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create List of all human protein encoding genes (based on uniprot proteins).')
	parser.add_argument('--HumanIDMap',help='Uniprot to Entrez File',default=currentDir+'HUMAN_9606_idmapping2021.dat')
	parser.add_argument('--GeneInfoFile',help='File listing all Human Genes from NCBI',default=currentDir+'Homo_sapiens_2021.gene_info')
	parser.add_argument('--OutputFile',help='File to write all Gene Encoding Proteins To.',default=parentDir+'DataGeneIDs2021.tsv')
	args = parser.parser_args()

	createEntrezUniprotMapping(args.HumanIDMap,args.GeneInfoFile,args.OutputFile)