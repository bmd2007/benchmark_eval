import os
import sys
#add parent and grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
currentDir = currentdir+'/'
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentDir = parentdir+'/'
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)

import requests
import re
import PPIPUtils
import urllib
import urllib.request
import time
import gzip
import shutil
def genAllFasta():
	uniprotFastaLoc, uniprotFastaLoc2 = PPIPUtils.getUniprotFastaLocations()

	nonYeastProteins = set()
	nonYeastFastas = set()
	allPairs = []
	allDIPProts = {}
	#load DIP protein names
	for fname in [currentDir+'NegativePairs.txt',currentDir+'PositivePairs.txt']:
		f = open(fname)
		print(fname)
		h = None
		for line in f:
			line = line.strip()
			if len(line) ==0:
				continue
			if h is None: #skip header
				h = line
				continue
			line = line.split('-')
			for i in range(0,2):
				allDIPProts[line[i]] = set()
			if fname == currentDir+'NegativePairs.txt':
				classVal = '0'
			else:
				classVal = '1'
			allPairs.append((line[0],line[1],classVal))
			
		f.close()

	if not os.path.isfile(parentDir+'Human2021/ProteinSetCreation/HUMAN_9606_idmapping2021.dat'):
		PPIPUtils.downloadFile('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz', parentDir+'Human2021/ProteinSetCreation/HUMAN_9606_idmapping2021.dat'+'.gz')
		with gzip.open(parentDir+'Human2021/ProteinSetCreation/HUMAN_9606_idmapping2021.dat'+'.gz', 'rb') as f_in:
			with open(parentDir+'Human2021/ProteinSetCreation/HUMAN_9606_idmapping2021.dat', 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)


	if not os.path.isfile(currentDir+'YEAST_559292_idmapping.dat'):
		PPIPUtils.downloadFile('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/YEAST_559292_idmapping.dat.gz', currentDir+'YEAST_559292_idmapping.dat'+'.gz')
		with gzip.open(currentDir+'YEAST_559292_idmapping.dat'+'.gz', 'rb') as f_in:
			with open(currentDir+'YEAST_559292_idmapping.dat', 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)


	#map DIP to uniprot
	#try human and yeast proteins
	for fname in [currentDir+'YEAST_559292_idmapping.dat',parentDir+'Human2021/ProteinSetCreation/HUMAN_9606_idmapping2021.dat']:
		f = open(fname)
		for line in f:
			line = line.strip()
			if len(line) == 0:
				continue
			line = line.split()
			if line[1] == 'DIP':
				uniprotName = line[0]
				dipName = line[2].split('-')[1]
				if dipName in allDIPProts:
					allDIPProts[dipName].add(uniprotName)
					if fname != 'YEAST_559292_idmapping.dat':
						print('Human',dipName,uniprotName)
						nonYeastProteins.add(dipName)
						
	uniprotRegex = re.compile(r'https://uniprot.org/uniprot/([^\']+)')
	speciesRegex = re.compile('Organism</font>\s*</th>\s*<td[^>]+>\s*([^<]+)')
	for item in allDIPProts:
		if len(allDIPProts[item]) == 0: #could not map DIP ID from uniprot, need to get it from DIP
			dipName = item
			dipNumber = str(int(item[:-1]))
			try: 
				url = "https://dip.doe-mbi.ucla.edu/dip/DIPview.cgi?PK="+dipNumber
				#need to use requests to handle dip cookies
				#note, we should download an SSL certificate manager and download the certificate for the site instead of turning off verification
				#may fix later
				f = requests.get(url,verify=False) 
				resp = f.text
				uniprotID = uniprotRegex.search(resp).group(1)
				allDIPProts[item].add(uniprotID)
				print('downloaded',item)
				speciesName = speciesRegex.search(resp)
				if speciesName:
					speciesName = speciesName.group(1).strip()
				if speciesName != 'Saccharomyces cerevisiae':
					print(speciesName,item)
					nonYeastProteins.add(item)
				time.sleep(5) #always wait after downloading a webpage to not flood the website
			except Exception as ex:#can't find sequence, so skip it
				print(ex)
				print(item,'missing')
				time.sleep(5) #always wait after downloading a webpage to not flood the website
				continue




	#get fasta per DIP ID
	allUni = set()
	for item in allDIPProts:
		allUni = allUni | allDIPProts[item]
		
	allFastas = PPIPUtils.parseUniprotFasta(uniprotFastaLoc, allUni)
	allFastas2 = PPIPUtils.parseUniprotFasta(uniprotFastaLoc2, allUni)
	allFastas.update(allFastas2)
	foundDIPFastas = set()
	f = open(currentDir+'allSeqs.fasta','w')
	for DIPNumber in allDIPProts:
		curSt = ''
		for item in allDIPProts[DIPNumber]:
			newSt = allFastas.get(item,'')
			if newSt == '':
				#if we can't find the sequence, in the tremble or sprot files, I doubt we will find it online.  But, we can try. . .
				try: 
					url = "https://www.uniprot.org/uniprot/"+item+".fasta"
					lst = urllib.request.urlopen(url).read().decode('utf-8').split('\n')
					newSt = ''.join(lst[1:])
					print('downloaded',item)
					time.sleep(5) #always wait after downloading a webpage to not flood the website
				except Exception as ex:#can't find sequence, so skip it
					print(ex)
					print(item,'missing')
					continue
			if len(newSt) > len(curSt):
				curSt = newSt
		if len(curSt) <=30:
			print(DIPNumber,allDIPProts[DIPNumber],curSt)
			continue
		f.write('>'+DIPNumber+'\n')
		f.write(curSt+'\n')
		foundDIPFastas.add(DIPNumber)
		if DIPNumber in nonYeastProteins:
			nonYeastFastas.add(DIPNumber)
	f.close()

	f = open(currentDir+'allPairs.tsv','w')
	for item in allPairs:
		if item[0] in foundDIPFastas and item[1] in foundDIPFastas:
			f.write('\t'.join(item) + '\n')
	f.close()

	f = open(currentDir+'NonYeastProteins.txt','w')
	f.write('\n'.join(nonYeastFastas))
	f.close()

	f = open(currentDir+'NonYeastPairs.tsv','w')
	for item in allPairs:
		if item[0] in foundDIPFastas and item[1] in foundDIPFastas:
			if item[0] in nonYeastFastas or item[1] in nonYeastFastas:
				f.write('\t'.join(item) + '\n')
	f.close()


	f = open(currentDir+'OnlyYeastPairs.tsv','w')
	for item in allPairs:
		if item[0] in foundDIPFastas and item[1] in foundDIPFastas:
			if item[0] not in nonYeastFastas and item[1] not in nonYeastFastas:
				f.write('\t'.join(item) + '\n')
	f.close()


if __name__ == '__main__':
	genAllFasta()