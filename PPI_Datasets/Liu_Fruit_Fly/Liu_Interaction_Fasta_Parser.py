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
import gzip
import shutil


def genAllFasta():
	uniprotFastaLoc, uniprotFastaLoc2 = PPIPUtils.getUniprotFastaLocations()



	f = open(currentDir+'Liu_Interactions.tsv')
	geneMap = {}
	intLst = []
	header = None
	for line in f:
		line = line.strip().split('\t')[1:]
		if header is None:
			header = line
			continue
		if line[2] == '2':
			line[2] = '0'
		intLst.append(line)
		for i in range(0,2):
			geneMap[line[i]] = set()
	f.close()

	protSet = set()
	
	
	
	if not os.path.isfile(currentDir+'DROME_7227_idmapping.dat'):
		PPIPUtils.downloadFile('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/DROME_7227_idmapping.dat.gz', currentDir+'DROME_7227_idmapping.dat'+'.gz')
		with gzip.open(currentDir+'DROME_7227_idmapping.dat'+'.gz', 'rb') as f_in:
			with open(currentDir+'DROME_7227_idmapping.dat', 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
				
				
	f = open(currentDir+'DROME_7227_idmapping.dat')
	for line in f:
		line = line.strip().split('\t')
		if line[1] in ['Gene_Synonym','Gene_ORFName']:
			if line[2] in geneMap:
				protSet.add(line[0])
				geneMap[line[2]].add(line[0])
	f.close()
				
				



	allFastas = PPIPUtils.parseUniprotFasta(uniprotFastaLoc, protSet)
	allFastas2 = PPIPUtils.parseUniprotFasta(uniprotFastaLoc2, protSet)
	allFastas.update(allFastas2)
	finalGenes = set()
	f = open(currentDir+'allSeqs.fasta','w')
	for gene in geneMap:
		curSt = ''
		newSt = ''
		for prot in geneMap[gene]:
			newSt = allFastas.get(prot,'')
			if newSt == '':
				#if we can't find the sequence, in the tremble or sprot files, I doubt we will find it online.  But, we can try. . .
				try: 
					url = "https://www.uniprot.org/uniprot/"+prot+".fasta"
					lst = urllib.request.urlopen(url).read().decode('utf-8').split('\n')
					newSt = ''.join(lst[1:])
					allFastas[prot] = newSt
					print('downloaded',prot)
					time.sleep(1) #always wait after downloading a webpage to not flood the website
				except Exception as ex:#can't find sequence, so skip it
					print(ex)
					print(prot,'missing')
					continue
			
			#take longest sequence
			if len(newSt) > len(curSt):
				curSt = newSt
					
		if len(curSt) <=30:
			print(gene,curSt,'\n',newSt)
			continue
		f.write('>'+gene+'\n')
		f.write(curSt+'\n')
		finalGenes.add(gene)
	f.close()
			
	f = open(currentDir+'pairLst.tsv','w')
	for item in intLst:
		if item[0] in finalGenes and item[1] in finalGenes:
			f.write('\t'.join(item)+'\n')
	f.close()


if __name__ == '__main__':
	genAllFasta()