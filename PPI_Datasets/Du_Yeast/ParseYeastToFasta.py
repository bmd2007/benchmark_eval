import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
currentDir = currentdir+'/'
import PPIPUtils
import urllib
import urllib.request
import time

def genAllFasta():
	f = open(currentDir+'Supplementary S1.csv')
	data = []
	proteins = set()
	header = None
	for line in f:
		if header is None:
			header = line
			continue
		line = line.strip().split(',')
		data.append(line)
		for i in range(0,2):
			proteins.add(line[i])
	f.close()

	uniprotFastaLoc, uniprotFastaLoc2 = PPIPUtils.getUniprotFastaLocations()


	allFastas = PPIPUtils.parseUniprotFasta(uniprotFastaLoc, proteins)
	allFastas2 = PPIPUtils.parseUniprotFasta(uniprotFastaLoc2, proteins)
	allFastas.update(allFastas2)
	finalProts = set()
	f = open(currentDir+'allSeqs.fasta','w')
	for prot in proteins:
		newSt = allFastas.get(prot,'')
		if newSt == '':
			#if we can't find the sequence, in the tremble or sprot files, I doubt we will find it online.  But, we can try. . .
			try: 
				url = "https://www.uniprot.org/uniprot/"+prot+".fasta"
				lst = urllib.request.urlopen(url).read().decode('utf-8').split('\n')
				newSt = ''.join(lst[1:])
				print('downloaded',prot)
				time.sleep(1) #always wait after downloading a webpage to not flood the website
			except Exception as ex:#can't find sequence, so skip it
				print(ex)
				print(prot,'missing')
				continue
		if len(newSt) <=30:
			print(prot,newSt)
			continue
		f.write('>'+prot+'\n')
		f.write(newSt+'\n')
		finalProts.add(prot)
	f.close()
		
	f = open(currentDir+'pairLst.tsv','w')
	for item in data:
		if item[0] in finalProts and item[1] in finalProts:
			f.write('\t'.join(item)+'\n')
	f.close()

if __name__ == '__main__':
	genAllFasta()
	