import sys
import os
currentDir = os.path.dirname(os.path.realpath(__file__))
parentDir = os.path.dirname(currentDir)+'/'
currentDir += '/'
sys.path.append(parentDir)
import PPIPUtils
import PairwisePreprocessUtils
from GeneOntologySemScorer import GeneOntologySemScorer

humanUniprotMapping = PairwisePreprocessUtils.createUniprotMapping(parentDir+'ppi_datasets/human2021/HumanUniprotEntrezProteinLst.tsv')

entrezIds = set()
for item in humanUniprotMapping:
	for item2 in humanUniprotMapping[item]:
		entrezIDs.add(item2)

print('Entrez',len(entrezIds))
entrezIds = sorted(entrezIds)


goFolder = parentDir+'ppi_datasets/human2021/GeneOntologyAggs/'
goOutputFolder=parentDir+'ppi_datasets/human2021/GeneOntologyComputed/'
scorer = GeneOntologySemScorer(goFolder,True,humanUniprotMapping)

PPIPUtils.mkdir(goOutputFolder)
fileDict = {}

for i in range(0,len(entrezIds)):
	for j in range(i,len(entrezIds)):
		d = {}
		sc.scoreProteinPair(entrezIds[i],entrezIds[j],d)
	if i == j: #first column
		if i == 0: #first row
			for item in d:
				fileDict[item] = open(goFolder+item+'.tsv','w')
				fileDict[item].write('\t'.join(str(s) for s in entrezIds))
		for item in d:
			fileDict[item].write('\n'+str(entrezIds[i]))
	for item in d:
		fileDict[item].write('\t'+str(d[item]))
for item in fileDict:
	fileDict[item].close()