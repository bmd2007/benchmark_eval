import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

createAllDatasets = False

#This section is optional, our source code contains
#copies of all the pairs/sequences used in dataset in our paper
#this is the code we ran to create those datasets, and will
#re-create them if run.  Note, the main Human Dataset may
#come out differently if using this code, as we randomly
#split from all possible proteins, and this code will download
#the newest data for proteins, which may not match with our original data
if createAllDatasets:
	#Create fasta/train test files for all previously used datasets
	#Du Yeast
	import PPI_Datasets.Du_Yeast.ParseYeastToFasta
	PPI_Datasets.Du_Yeast.ParseYeastToFasta.genAllFasta()

	#Guo Yeast (From Chen)
	import PPI_Datasets.Guo_Data_Yeast_Chen.GenFasta
	PPI_Datasets.Guo_Data_Yeast_Chen.GenFasta.genAllFasta()

	#Guo Yeast (From Tian)
	import PPI_Datasets.Guo_Data_Yeast_Tian.GuoToIndexAndFasta
	PPI_Datasets.Guo_Data_Yeast_Tian.GuoToIndexAndFasta.genAllFasta()

	#Guo Multispecies (From Chen)
	import PPI_Datasets.Guo_MultiSpecies_Chen.DictionaryToFasta
	PPI_Datasets.Guo_MultiSpecies_Chen.DictionaryToFasta.genAllFasta()

	#Jia Yeast Data
	import PPI_Datasets.Jia_Data_Yeast.JiaDataYeastParser
	PPI_Datasets.Jia_Data_Yeast.JiaDataYeastParser.genAllFasta()

	#Li Alzheimer's Data
	import PPI_Datasets.Li_AD.LiToIndexAndFasta
	PPI_Datasets.Li_AD.LiToIndexAndFasta.genAllFasta()

	#Liu Fruit Fly Data
	import PPI_Datasets.Liu_Fruit_Fly.Liu_Interaction_Fasta_Parser
	PPI_Datasets.Liu_Fruit_Fly.Liu_Interaction_Fasta_Parser.genAllFasta()

	#Martin's H.Pylori Data
	import PPI_Datasets.Martin_H_pylori.MartinDataCreator
	PPI_Datasets.Martin_H_pylori.MartinDataCreator.genAllFasta()

	#Pan's Full Human Data, Filtered Human Data, and Martin's Human Data
	import PPI_Datasets.Pan_Human_Data.BuildPanData
	PPI_Datasets.Pan_Human_Data.BuildPanData.genAllFasta()

	#Richoux's strict Human Data (C2)
	import PPI_Datasets.Richoux_Human_Data.RichouxToIndexAndFasta
	PPI_Datasets.Richoux_Human_Data.RichouxToIndexAndFasta.genAllFasta()

	#Generate the data for our human datasets
	#get protein/gene list
	import PPI_Datasets.Human2021.ProteinSetCreation.EntrezUniprotCreator
	PPI_Datasets.Human2021.ProteinSetCreation.EntrezUniprotCreator.createEntrezUniprotMapping()

	#get fastas for all proteins/genes
	import PPI_Datasets.Human2021.HumanToFasta
	PPI_Datasets.Human2021.HumanToFasta.genAllFasta()

	#download/parse BioGRID data
	import PPI_Datasets.Human2021.BioGRID2021.biogrid_psi_parser_all
	PPI_Datasets.Human2021.BioGRID2021.biogrid_psi_parser_all.getHumanFromBioGRID()

	#Filter BioGRID interactions down to human interactions from genes that map to uniprot IDs, aren't RNA binding, and aren't obsolete in Entrez
	import PPI_Datasets.Human2021.BioGRID2021.PPISetCreator
	PPI_Datasets.Human2021.BioGRID2021.PPISetCreator.createPPISet()

	#Filters out Entrez/Proteins that were not kept during fasta generation (short sequences), creating a final interaction list
	import PPI_Datasets.Human2021.BioGRID2021.BioGRIDPPICreator
	PPI_Datasets.Human2021.BioGRID2021.BioGRIDPPICreator.createPPIListFromBioGRIDandFasta()

	#Create (5+21)*2 train sets and (5+21*2)*2 test sets for randomly drawn and held out protein pairs
	#may be different than the dataset used in our paper depending on downloaded  files and random seed
	import PPI_Datasets.Human2021.BioGRID2021.CreateDatasets
	PPI_Datasets.Human2021.BioGRID2021.CreateDatasets.createDataSplits()

#gen sequence features
import RunTests
RunTests.genSequenceFeatures()


#gen pairwise feature for human data
#Note, this function computes semantic similarity using a gpu by default, but can be switched to cpu memory editing lines 
#50-51 to use 'cpu' instead of 'cuda', without much of a difference in processing time.  
#Regardless, at least 24gb of free memory will be needed to calculate the values for the regular and descending semantic similarity 
#matrices using biological process data on annotations used by human proteins.
import PPI_Datasets.Human2021.PairwisePreProcessHumanData
PPI_Datasets.Human2021.PairwisePreProcessHumanData.genPairwiseHumanData()


#gen sequence features
import RunTests
RunTests.RunAll()





