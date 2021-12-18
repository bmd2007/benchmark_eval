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


def createEntrezEnsemblMapping(GeneToEnsembleFile=currentDir+'gene2ensembl.gz',humanGeneListFile=parentDir+'HumanUniprotEntrezProteinLst.tsv',outputFile=parentDir+'EntrezEnsembl2021.tsv',taxID='9606'):
	if not os.path.isfile(GeneToEnsembleFile):
		PPIPUtils.downloadFile('https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2ensembl.gz', GeneToEnsembleFile)
				


	data = PPIPUtils.parseTSV(humanGeneListFile)
	header = {}
	for item in data[0]:
		header[item] = len(header)
	data = data[1:]
	entrezSet = set()
	for line in data:
		entrezSet.add(line[header['Entrez']])
	
	
	f = gzip.open(GeneToEnsembleFile,'rb')
	f2 = open(outputFile,'w')
	f2.write('EntrezID\tEnsemblGeneID\tEnsemblProteinID\tEnsemblRNAID')
	header = {}
	for line in f:
		line = line.decode('utf-8').strip().split()
		if len(header) == 0:
			for item in line:
				header[item] = len(header)
			continue
		tID = line[header['#tax_id']]
		geneID = line[header['GeneID']]
		ensemblGID = line[header['Ensembl_gene_identifier']]
		ensemblRID = line[header['Ensembl_rna_identifier']]
		ensemblPID = line[header['Ensembl_protein_identifier']]
		if str(taxID) != tID:
			continue
		if geneID not in entrezSet:
			continue
		f2.write('\n'+geneID+'\t'+ensemblGID+'\t'+ensemblPID+'\t'+ensemblRID)
	f2.close()
	
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create a Mapping of EntrezIDs to Ensembl.')
	
	
	parser.add_argument('--GeneToEnsemblFile',help='Save location of gene2ensembl file from NCBI FTP website',default=currentDir+'gene2ensembl.gz')
	parser.add_argument('--HumanGeneListFile',help='File listing all Entrez IDs we are using',default=parentDir+'HumanUniprotEntrezProteinLst.tsv')
	parser.add_argument('--OutputFile',help='File to write all Gene To Ensembl Mapping to.',default=parentDir+'EntrezEnsembl2021.tsv')
	parser.add_argument('--TaxID',help='Tax ID to filter Entrez data by',default='9606')
	args = parser.parse_args()

	createEntrezEnsemblMapping(args.GeneToEnsemblFile,args.HumanGeneListFile,args.OutputFile,args.TaxID)