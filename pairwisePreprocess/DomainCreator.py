import gzip
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import PPIPUtils

#takes protein2ipr.dat.gz from interpro and parses out human data
#humanUnprotMapping can be a dictionary mapping uniprot terms to entrez genes
#or a set containing human uniprot terms (no mapping, but needed to filter out non-human data)
def parseInterProFile(interproGZFile,outputFileName,humanUniprotMapping):
	if not os.path.exists(interproGZFile):
		PPIPUtils.downloadFile('https://ftp.ebi.ac.uk/pub/databases/interpro/protein2ipr.dat.gz',interproGZFile)

	if type(humanUniprotMapping) is set:
		mapping=  False
	else:
		mapping = True
	
	domainMap = {}
	f = gzip.open(interproGZFile,'rb')
	for line in f:
		line = line.decode('utf-8')
		line = line.strip().split()
		prot = line[0]
		dom = line[1]
		if prot in humanUniprotMapping:
			protLst = [prot]
			if mapping:
				protLst = humanUniprotMapping[prot]
			
			for p in protLst:
				if p not in domainMap:
					domainMap[p] = set()
				domainMap[p].add(dom)
	f.close()

	f = open(outputFileName,'w')
	f.write('Protein\tDomains\n')
	for p, dLst in domainMap.items():
		for dom in dLst:
			f.write(p+'\t'+dom+'\n')
	f.close()
	

#parses file from pfam, Pfam-A.regions.tsv.gz
def parsePFamToProteinFile(pfamGZFile,outputFileName,humanUniprotMapping):
	if not os.path.exists(pfamGZFile):
		PPIPUtils.downloadWGet('http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.regions.tsv.gz',pfamGZFile)

	if type(humanUniprotMapping) is set:
		mapping=  False
	else:
		mapping = True
	
	domMap = {}
	f = gzip.open(pfamGZFile,'rb')
	header = None
	for line in f:
		line = line.decode('utf-8')
		line = line.strip().split()
		if header is None:
			header = {}
			for i in range(0,len(line)):
				header[line[i]] = i
			continue
		prot = line[header['pfamseq_acc']]
		dom = line[header['pfamA_acc']]
		if prot in humanUniprotMapping:
			protLst = [prot]
			if mapping:
				protLst = humanUniprotMapping[prot]
			
			for p in protLst:
				if p not in domMap:
					domMap[p] = set()
				domMap[p].add(dom)
	f.close()

	f = open(outputFileName,'w')
	f.write('Protein\tDomains\n')
	for p, dLst in domMap.items():
		for dom in dLst:
			f.write(p+'\t'+dom+'\n')
	f.close()
		
def createUniprotNameMapping(uniprotDatMappingFile):
	f = open(uniprotDatMappingFile)
	#grab names for all uniprotIDs
	
	nameMapping = {}
	for line in f:
		line = line.strip().split()
		if line[1] == 'UniProtKB-ID':
			if line[2] not in nameMapping:
				nameMapping[line[2]] = set()
			nameMapping[line[2]].add(line[0])
		elif line[1] == 'Gene_Synonym':
			line[2] = line[2]+'_HUMAN'
			if line[2] not in nameMapping:
				nameMapping[line[2]] = set()
			nameMapping[line[2]].add(line[0])
	f.close()
	return nameMapping
	

#takes prints42_0.dat as printsFile
def parsePrintsToProteinsFile(printsFile,humanUniprotMapping,uniprotNameMapping,hcFile,lcFile):
	if not os.path.exists(printsFile):
		PPIPUtils.downloadFile('https://ftp.ebi.ac.uk/pub/databases/prints/prints42_0.dat.gz',printsFile)
	if type(humanUniprotMapping) is set:
		mapping = False
	else:
		mapping = True
	f = gzip.open(printsFile)
	#we are going to parse 3 types of lines
	#tp;  True positive matches, either uniprot IDs or names
	#st;  Partial matches, which we will record as low confidence matches
	#gc;  Name of prints match, used to mark start of new item
	
	curName = None
	highConf = set()
	lowConf = set()
	for line in f:
		try:
			line = line.decode('utf-8')
		except:
			print('Error',line)
			continue
		line = line.strip().split()
		if line[0] == 'gc;':
			curName = ' '.join(line[1:]) #uncertain if any names have spaces, but this should work either way
		elif line[0] in ['st;','tp;']:
			for item in line[1:]:
				name = item
				uniNameLst = []
#				if len(item.split('_')) > 1 and item.split('_')[1] == 'HUMAN':
#					print(item, item in uniprotNameMapping, uniprotNameMapping.get(item,set()))
				#if this is a name, and it names to our name list, then add uniprot proteins mapping to name
				if item in uniprotNameMapping:
					uniNameLst = list(uniprotNameMapping)
				else: #otherwise, assume uniprot protein id, and add it to list
					uniNameLst = [item]
				protLst = []
				for uniName in uniNameLst:
					if uniName in humanUniprotMapping:
						#if mapping to genes or some other concept, grab concepts from list
						if mapping:
							protLst.extend(humanUniprotMapping[uniName])
						else: #otherwise, append uniprot name to list
							protLst.append(uniName)
				for prot in protLst:
					if line[0] == 'tp;':#if high confidence, add to high confidence dict
						highConf.add((prot,curName))
					lowConf.add((prot,curName))
	f.close()
	f = open(hcFile,'w')
	f.write('Prot\tPrints\n')
	for item in highConf:
		f.write('\t'.join(item)+'\n')
	f.close()
	f = open(lcFile,'w')
	f.write('Prot\tPrints\n')
	for item in lowConf:
		f.write('\t'.join(item)+'\n')
	f.close()
	
#takes prosite_alignments.tar.gz as the prositeFile
def PrositeToProteinFile(prositeGZFile,humanUniprotMapping,outputFileName):
	if not os.path.exists(prositeGZFile):
		PPIPUtils.downloadFile('https://ftp.expasy.org/databases/prosite/prosite_alignments.tar.gz',prositeGZFile)

	if type(humanUniprotMapping) is set:
		mapping = False
	else:
		mapping = True

	#Only need lines mapping protise to protein
	#format for mapping lines
	#>CYH2_MOUSE|P63034/54-241: SEC7|PS50190/46.8
	f = gzip.open(prositeGZFile,'rb')
	prositeMappings = set()
	header = None
	for line in f:
		line = line.decode('utf-8')
		if line[0] != '>':
			continue
		line = line.strip().split('|')
		protName = line[1].split('/')[0]
		prositeName = line[2].split('/')[0]
		if protName in humanUniprotMapping:
			if mapping:
				for item in humanUniprotMapping[protName]:
					prositeMappings.add((item,prositeName))
			else:
				prositeMappings.add((protName,prositeName))
	f.close()
	
	f = open(outputFileName,'w')
	f.write('ProtName\tPrositeName\n')
	for item in prositeMappings:
		f.write('\t'.join(item)+'\n')
	f.close()
	
	
