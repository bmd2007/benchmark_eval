import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
currentDir = currentdir + '/'
import PPIPUtils
import urllib
import urllib.request
import time
import argparse


def genAllFasta(geneIDFile = currentDir+'DataGeneIDs2021.tsv', entrezProteinOutputFile = currentDir + 'HumanUniprotEntrezProteinLst.tsv',ProteinMappingOutputFile=currentDir+'ProteinMapping.tsv',createFastas=True):
	uniprotFastaLoc, uniprotFastaLoc2 = PPIPUtils.getUniprotFastaLocations()
	allGenesLst = []
	allUni = set()
	idMapping ={}
	f = open(geneIDFile)
	for line in f:
		line = line.strip().split('\t')
		allGenesLst.append(line)
		uni = line[4]
		entrez = line[0]
		if entrez not in idMapping:
			idMapping[entrez] = set()
		idMapping[entrez].add(uni)
		allUni.add(uni)
	f.close()

	if createFastas:
		allFastas = PPIPUtils.parseUniprotFasta(uniprotFastaLoc, allUni)
		allFastas2 = PPIPUtils.parseUniprotFasta(uniprotFastaLoc2, allUni)
		allFastas.update(allFastas2)
		f = open(currentDir+'allSeqs.fasta','w')
		idx = 0
		for entrez in idMapping:
			curSt = ''
			for item in idMapping[entrez]:
				newSt = allFastas.get(item,'')
				if newSt == '':
					#if we can't find the sequence, in the tremble or sprot files, I doubt we will find it online.  But, we can try. . .
					try: 
						url = "https://www.uniprot.org/uniprot/"+item+".fasta"
						lst = urllib.request.urlopen(url).read().decode('utf-8').split('\n')
						newSt = ''.join(lst[1:])
						print('downloaded',item)
						time.sleep(1) #always wait after downloading a webpage to not flood the website
					except Exception as ex:#can't find sequence, so skip it
						print(ex)
						print(item,'missing')
						continue
				if len(newSt) > len(curSt):
					curSt = newSt
			if len(curSt) <=30:
				print(entrez,idMapping[entrez],curSt)
				continue
			f.write('>'+entrez+'\n')
			f.write(curSt+'\n')
		f.close()
		

	f = open(currentDir+'allSeqs.fasta')
	entrezSet = set()
	idx = 0
	for line in f:
		if idx == 0:
			entrezSet.add(line.strip().strip('>'))
		idx += 1
		idx = idx % 2
	f.close()

	f = open(geneIDFile)
	f2 = open(entrezProteinOutputFile,'w')
	f3 = open(ProteinMappingOutputFile,'w')
	f2.write('Entrez\tSymbol\tUniprot\n')
	h = []
	for line in f:
		line = line.strip().split('\t')
		if len(h) == 0: #skip header
			h = line
			continue
		if line[0] not in entrezSet:
			print('skipping',line[0])
		else:
			f2.write(line[0] + '\t' + line[1] + '\t' + line[4] + '\n')
			f3.write(line[0]+'\t'+line[4]+'\n')
	f.close()
	f2.close()
	f3.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parses BioPhysical interactions from BioGRID to create a final output file')
	parser.add_argument('--GeneIDFile',help='File to read Interactions from',default=currentDir+'DataGeneIDs2021.tsv')
	parser.add_argument('--EntrezProteinOutputFile',help='File to read proteins from',default=currentDir+'HumanUniprotEntrezProteinLst.tsv')
	parser.add_argument('--ProteinMappingOutputFile',help='Folder to save new datasets into',default=currentDir+'ProteinMapping.tsv')
	parser.add_argument('--createFastas',help='Whether to (re)generate allfastas file',default='True')
	args = parser.parser_args()

	cFastas = True if args.createFastas.upper() not in ['F','FALSE'] else False
	genAllFasta(args.GeneIDFile,args.EntrezProteinOutputFile,args.ProteinMappingOutputFile,cFastas)
	