import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
preprocessDir = parentdir2+'/pairwisePreprocess/'
sys.path.append(parentdir)
sys.path.append(parentdir2)
sys.path.append(preprocessDir)
currentDir = currentdir+'/'
parentDir = parentdir+'/'
parentDir2 = parentdir2+'/'
import PPIPUtils
import GeneOntologyCreator
from GenericTermPairScorer import GenericTermPairScorer
from GOTermPairScorer import GOTermPairScorer
from SortedMatrixParser import SortedMatrixParser
from SortedMatrixParserIOU import SortedMatrixParserIOU
import PairwisePreprocessUtils
import DomainCreator
from GeneOntologySemScorer import GeneOntologySemScorer
import PPIPUtils
import gzip
import shutil

def genPairwiseHumanData():
	PPIPUtils.makeDir(currentDir+'HumanAnnotationData/')
	domainFolder = currentDir+'DomainAggs/'
	PPIPUtils.makeDir(domainFolder)
	#create the data files need mapping human proteins to domains and gene ontology terms, as well as the needed data to calculate semantec similiarity
	#map uniprot to entrez
	humanUniprotMapping = PairwisePreprocessUtils.createUniprotMapping(currentDir+'HumanUniprotEntrezProteinLst.tsv')
	humanUniprotNameMapping = DomainCreator.createUniprotNameMapping(currentDir+'ProteinSetCreation/HUMAN_9606_idmapping2021.dat')
	

	DomainCreator.parseInterProFile(currentDir+'HumanAnnotationData/protein2ipr.dat.gz',domainFolder+'HumanGeneInterPro.tsv',humanUniprotMapping)
	DomainCreator.parsePFamToProteinFile(currentDir+'HumanAnnotationData/Pfam-A.regions.tsv.gz',domainFolder+'PfamProteins.tsv',humanUniprotMapping)
	DomainCreator.PrositeToProteinFile(currentDir+'HumanAnnotationData/prosite_alignments.tar.gz',humanUniprotMapping,domainFolder+'PrositeProteins.tsv')

	if not os.path.exists(currentDir +'HumanAnnotationData/go-basic2021.obo'):
		PPIPUtils.downloadFile('http://purl.obolibrary.org/obo/go/go-basic.obo', currentDir +'HumanAnnotationData/go-basic2021.obo')
	if not os.path.exists(currentDir +'HumanAnnotationData/goa_human2021.gaf'):
		PPIPUtils.downloadFile('http://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz', currentDir +'HumanAnnotationData/goa_human2021.gaf.gz')
		with gzip.open(currentDir +'HumanAnnotationData/goa_human2021.gaf.gz', 'rb') as f_in:
			with open(currentDir +'HumanAnnotationData/goa_human2021.gaf', 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
	PPIPUtils.makeDir(currentDir+'GeneOntologyAggs/')
	
	GeneOntologyCreator.geneOntologyL2Creator(currentDir +'HumanAnnotationData/go-basic2021.obo',currentDir +'HumanAnnotationData/goa_human2021.gaf',currentDir+'GeneOntologyAggs/',humanUniprotMapping,'cuda')
	GeneOntologyCreator.geneOntologySSCreator(currentDir +'HumanAnnotationData/go-basic2021.obo',currentDir +'HumanAnnotationData/goa_human2021.gaf',currentDir+'GeneOntologyAggs/',True,'cuda')
	

		

	#list of known human protein interactions, as entrez IDs
	interactionLst = PPIPUtils.parseTSV(currentDir+'BioGRID2021/HumanUniprotEntrezInteractionLst.tsv')
	interactionSet = set()
	for item in interactionLst:
		item = tuple(sorted(item))
		interactionSet.add(item)
	interactionLst = interactionSet

	#get list of interactions to be held out for each protein group
	groups = [set(),set(),set(),set(),set(),set()]
	groupData = PPIPUtils.parseTSV(currentDir+'BioGRID2021/GroupData.tsv')
	for item in groupData[1:]:
		groups[int(item[0])].add(item[1])
		
	exclusionLst = []
	for i in range(0,6):
		for j in range(i,6):
			curLst = set()
			for item in interactionLst:
				if item[0] in groups[i] or item[0] in groups[j] or item[1] in groups[i] or item[1] in groups[j]:
					curLst.add(item)
			exclusionLst.append(curLst)
			

	#perform calculations for different pairwise data

	#start with loading datasets
	#load all train and test datasets
	#we need all pairs to be sorted, and all datasets to be sorted, in the same order we are sorting matrix based files in, so we can do out of memory calculations
	def parseProteinPairFile(fname,folderName,outName):
		data = PPIPUtils.parseTSV(fname)
		pairs = [sorted(x[0:2]) for x in data] #need all pairs to be sorted in the same order we are saving matrix files
		pairs.sort()
		#need positive interaction data for some files (where positive pairs are excluded)
		posPairs = []
		for line in data:
			if line[2] == '1':
				posPairs.append(tuple(sorted(line[0:2])))
		PPIPUtils.makeDir(folderName)
		allData = [sorted(x[0:2]) + [x[2]] for x in data]
		allData.sort()
		PPIPUtils.writeTSV2DLst(folderName+outName,allData)
		return pairs, set(posPairs)
		

	r50Trains = []
	r50Tests = []
	r20Trains1 = []
	r20Trains2 = []
	r20Tests1 = []
	r20Tests2 = []
	h50Trains = []
	h50Tests = []
	h20Trains1 = []
	h20Trains2 = []
	h20Tests1 = []
	h20Tests2 = []

	outputDirRoot = currentDir+'BioGRID2021/PairwiseDatasets/'
	PPIPUtils.makeDir(outputDirRoot)
	for i in range(0,5):
		r50Trains.append(parseProteinPairFile(currentDir+'BioGRID2021/Random50/Train_'+str(i)+'.tsv',outputDirRoot+'r50_'+str(i)+'/','train.tsv'))
		r50Tests.append(parseProteinPairFile(currentDir+'BioGRID2021/Random50/Test_'+str(i)+'.tsv',outputDirRoot+'r50_'+str(i)+'/','test.tsv'))
		r20Trains1.append(parseProteinPairFile(currentDir+'BioGRID2021/Random20/Train_'+str(i)+'.tsv',outputDirRoot+'r20_1_'+str(i)+'/','train.tsv'))
		r20Trains2.append(parseProteinPairFile(currentDir+'BioGRID2021/Random20/Train_'+str(i)+'.tsv',outputDirRoot+'r20_2_'+str(i)+'/','train.tsv'))
		r20Tests1.append(parseProteinPairFile(currentDir+'BioGRID2021/Random20/Test1_'+str(i)+'.tsv',outputDirRoot+'r20_1_'+str(i)+'/','test.tsv'))
		r20Tests2.append(parseProteinPairFile(currentDir+'BioGRID2021/Random20/Test2_'+str(i)+'.tsv',outputDirRoot+'r20_2_'+str(i)+'/','test.tsv'))


	for i in range(0,6):
		for j in range(i,6):
			h50Trains.append(parseProteinPairFile(currentDir+'BioGRID2021/HeldOut50/Train_'+str(i)+'_'+str(j)+'.tsv',outputDirRoot+'h50_'+str(i)+'_'+str(j)+'/','train.tsv'))
			h50Tests.append(parseProteinPairFile(currentDir+'BioGRID2021/HeldOut50/Test_'+str(i)+'_'+str(j)+'.tsv',outputDirRoot+'h50_'+str(i)+'_'+str(j)+'/','test.tsv'))
			h20Trains1.append(parseProteinPairFile(currentDir+'BioGRID2021/HeldOut20/Train_'+str(i)+'_'+str(j)+'.tsv',outputDirRoot+'h20_1_'+str(i)+'_'+str(j)+'/','train.tsv'))
			h20Trains2.append(parseProteinPairFile(currentDir+'BioGRID2021/HeldOut20/Train_'+str(i)+'_'+str(j)+'.tsv',outputDirRoot+'h20_2_'+str(i)+'_'+str(j)+'/','train.tsv'))
			h20Tests1.append(parseProteinPairFile(currentDir+'BioGRID2021/HeldOut20/Test1_'+str(i)+'_'+str(j)+'.tsv',outputDirRoot+'h20_1_'+str(i)+'_'+str(j)+'/','test.tsv'))
			h20Tests2.append(parseProteinPairFile(currentDir+'BioGRID2021/HeldOut20/Test2_'+str(i)+'_'+str(j)+'.tsv',outputDirRoot+'h20_2_'+str(i)+'_'+str(j)+'/','test.tsv'))
		
	#we now have the list of all the pairs for each dataset, all positive pairs in the test datasets, and all positive pairs for each held out group pair
	#we also output a sorted list for each dataset, as this is the order we will be outputting features in for pairwise data
	#now we can do feature computation

	#start with domain pair data and GO L2 Frequency Data
	domainFiles = {'PFam':domainFolder+'PfamProteins.tsv','Prosite':domainFolder+'PrositeProteins.tsv','InterPro':domainFolder+'HumanGeneInterPro.tsv'}
	#go folder 
	goFolder = currentDir+'GeneOntologyAggs/'



	def DomainGOScoring(trainData,testData,heldOutTest,fullOutputDir, heldoutGenePair = None):
		global domainFiles
		global goFolder
		global interactionLst
		domainScorers = []
		for name, fname in domainFiles.items():
			domainScorers.append(GenericTermPairScorer(PPIPUtils.parseTSV(fname),interactionLst,name+'_All_'))
			domainScorers.append(GenericTermPairScorer(PPIPUtils.parseTSV(fname),interactionLst-heldOutTest,name+'_Non_Test_'))
			if heldoutGenePair is not None:
				domainScorers.append(GenericTermPairScorer(PPIPUtils.parseTSV(fname),interactionLst-heldoutGenePair,name+'_Heldout_'))
		
		
		#train data
		PairwisePreprocessUtils.scorePairs(trainData,fullOutputDir+'train_domainPairs.tsv',domainScorers)
		
		#testData
		PairwisePreprocessUtils.scorePairs(testData,fullOutputDir+'test_domainPairs.tsv',domainScorers)
		
		#gene ontology data
		geneOntologyL2Scorers = []
		geneOntologyL2Scorers.append(GOTermPairScorer(goFolder+'L2Terms.tsv',interactionLst,'GOL2Freq_All_'))
		geneOntologyL2Scorers.append(GOTermPairScorer(goFolder+'L2Terms.tsv',interactionLst-heldOutTest,'GOL2Freq_Non_Test_'))
		if heldoutGenePair is not None:
			geneOntologyL2Scorers.append(GOTermPairScorer(goFolder+'L2Terms.tsv',interactionLst-heldoutGenePair,'GOL2_Freq'+'_Heldout_'))

		#train data
		PairwisePreprocessUtils.scorePairs(trainData,fullOutputDir+'train_GOL2Pairs.tsv',geneOntologyL2Scorers)
		
		#testData
		PairwisePreprocessUtils.scorePairs(testData,fullOutputDir+'test_GOL2Pairs.tsv',geneOntologyL2Scorers)


	#r50, r20 data 
	for i in range(5,5):
		#r50
		DomainGOScoring(r50Trains[i][0],r50Tests[i][0],r50Tests[i][1],outputDirRoot+'r50_'+str(i)+'/',None)
		#r20
		DomainGOScoring(r20Trains1[i][0],r20Tests1[i][0],r20Tests1[i][1],outputDirRoot+'r20_1_'+str(i)+'/',None)
		DomainGOScoring(r20Trains2[i][0],r20Tests2[i][0],r20Tests2[i][1],outputDirRoot+'r20_2_'+str(i)+'/',None)
		
	#h50, h20 data
	idx = -1
	for i in range(6,6):
		for j in range(i,6):
			idx += 1
			#h50
			DomainGOScoring(h50Trains[idx][0],h50Tests[idx][0],h50Tests[idx][1],outputDirRoot+'h50_'+str(i)+'_'+str(j)+'/',exclusionLst[idx])
			#h20
			DomainGOScoring(h20Trains1[idx][0],h20Tests1[idx][0],h20Tests1[idx][1],outputDirRoot+'h20_1_'+str(i)+'_'+str(j)+'/',exclusionLst[idx])
			DomainGOScoring(h20Trains2[idx][0],h20Tests2[idx][0],h20Tests2[idx][1],outputDirRoot+'h20_2_'+str(i)+'_'+str(j)+'/',exclusionLst[idx])
		

	#Sementic similarity and l2 overlap scoring
	#These scores are unaffected by the held out proteins/test data, so we no longer need to worry about anything but the pairs themselves

	folderLst = []
	#create a list of dataset folders
	for i in range(0,5):
		folderLst.append(outputDirRoot+'r50_'+str(i)+'/')
	for i in range(0,5):
		folderLst.append(outputDirRoot+'r20_1_'+str(i)+'/')
		folderLst.append(outputDirRoot+'r20_2_'+str(i)+'/')

	for i in range(0,6):
		for j in range(i,6):
			folderLst.append(outputDirRoot+'h50_'+str(i)+'_'+str(j)+'/')

	for i in range(0,6):
		for j in range(i,6):
			folderLst.append(outputDirRoot+'h20_1_'+str(i)+'_'+str(j)+'/')
			folderLst.append(outputDirRoot+'h20_2_'+str(i)+'_'+str(j)+'/')


			
	def addToDict(dictionary, data,idx):
		for pair in data:
			pair = tuple(pair)
			if pair not in dictionary:
				dictionary[pair] = set()
			dictionary[pair].add(idx)
		return dictionary

	#created a sorted list containing tuples of pairs, list of datasets
	pairDict= {}
	for i in range(0,5):
		addToDict(pairDict,r50Trains[i][0],i*2)
		addToDict(pairDict,r50Tests[i][0],i*2+1)
	x = 10
	for i in range(0,5):
		addToDict(pairDict,r20Trains1[i][0],x+i*4)
		addToDict(pairDict,r20Tests1[i][0],x+i*4+1)
		addToDict(pairDict,r20Trains2[i][0],x+i*4+2)
		addToDict(pairDict,r20Tests2[i][0],x+i*4+3)
	x = 30
	for i in range(0,21):
		addToDict(pairDict,h50Trains[i][0],x+i*2)
		addToDict(pairDict,h50Tests[i][0],x+i*2+1)
	x = 72
	for i in range(0,21):
		addToDict(pairDict,h20Trains1[i][0],x+i*4)
		addToDict(pairDict,h20Tests1[i][0],x+i*4+1)
		addToDict(pairDict,h20Trains2[i][0],x+i*4+2)
		addToDict(pairDict,h20Tests2[i][0],x+i*4+3)
		
	pairLst = []
	for item, s in pairDict.items():
		pairLst.append((item,s))
	pairLst.sort()


	#now we have a list mapping all pairs we need, to all file indices we need
	#calculate semantic similarities for all pairs

	def createFileLsts(folderLst,name):
		fileLst = []
		for item in folderLst:
			fileLst.append(open(item+'train_'+name+'.tsv','w'))
			fileLst.append(open(item+'test_'+name+'.tsv','w'))
		return fileLst

	fileLst = createFileLsts(folderLst,'GOSS')


	scorerLst = []
	scorerLst.append(GeneOntologySemScorer(goFolder,True,humanUniprotMapping))


	#score GO SS Data
	PairwisePreprocessUtils.scoreAllPairs(pairLst,scorerLst,fileLst)

	for f in fileLst:
		f.close()
		

	#score GOL2 intersection data
	fileLst = createFileLsts(folderLst,'GOL2Int')
	scorerLst = []
	for item in ['MF','CC','BP']:
		scorerLst.append(SortedMatrixParserIOU(goFolder+'L2Intersections_'+item+'.tsv',goFolder+'L2Unions_'+item+'.tsv','GOL2_'+item))

	PairwisePreprocessUtils.scoreAllPairs(pairLst,scorerLst,fileLst)

				


if __name__ == '__main__':
	genPairwiseHumanData()
