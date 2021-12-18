import os
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
import random
import copy
from DatasetGenerator import *
import argparse

#creates original datasets, including 5-fold cross validation with 50/50 samples, 5 sets of 80/20 train, 90/10 test, and 498.5/1.5 test for random tests (100k per train)
#21 folds of 50/50 train and test for held out validation (train =100k, test = max)
#21 folds 80/20 train (100k each) 90/10 test (test=max) and 99.7k/0.3k test (100k-200k per test)
def createDataSplits(interactionFileName=currentDir+'HumanUniprotEntrezInteractionLst.tsv',proteinDataFileName=parentDir+'HumanUniprotEntrezProteinLst.tsv',saveFolderName=currentDir,proteinNameFormat='int',seed=1):
	random.seed(1)

	if saveFolderName[-1] not in ['/','\\']:
		saveFolderName+='/'
			
	if proteinNameFormat == 'int':
		interactionLst = PPIPUtils.parseTSV(interactionFileName,'int')
	else:
		interactionLst = PPIPUtils.parseTSV(interactionFileName)
	protData = PPIPUtils.parseTSV(proteinDataFileName)
	proteinLst = set()
	for i in range(1,len(protData)):
		if proteinNameFormat == 'int':
			proteinLst.add(int(protData[i][0]))
		else:
			proteinLst.add(protData[i][0])
	proteinLst = list(proteinLst)

	#create 5 cross folds of random data, with 62500 positive and negative pairs, and save them to the folder Random50
	createRandomKFoldData(interactionLst, proteinLst,62500,62500,5,saveFolderName+'Random50/')
	#create 5 training groups, with 20000 positive and 80000 random pairs, with 5 tests of 10000/90000 and 5 tests of 3000/997000 pos/neg pairs, and save them to folder random20
	createRandomData(interactionLst, proteinLst, ((20000,80000,'Train_'),[(10000,90000,'Test1_'),(1500,498500,'Test2_')]),5,saveFolderName+'Random20/')

	#create 6 protein groups, and group all interactions into their group pairs, ensuring at least 3125 pairs per small group and 6250 per large group, with a maximum of 100 attempts
	protGroups, protGroupDict, protGroupInts = createGroupData(interactionLst,proteinLst,6,3125,6250,100)

	#save the protein groups to a file
	f = open(currentDir+'GroupData.tsv','w')
	f.write('Group ID\tProtein ID\n')
	for i in range(0,len(protGroups)):
		for item in protGroups[i]:
			f.write(str(i)+'\t'+str(item)+'\n')
	f.close()

	#print grid sizes
	print('Grid Interaction List Sizes')
	for i in range(0,len(protGroupInts)):
		for j in range(i,len(protGroupInts[i])):
			print(i,j,len(protGroupInts[i][j]))


	#create 21 group held-out validation using 50/50 splits
	#for each split, generate training data:
	#if held out data is small set, use 4000 pairs from large sets and 2000 pairs from small sets to generate 50,000 positive pairs from the 10 large and 5 small heldout sets
	#if held out data is large set, use 6250 pairs from large sets and 3125 pairs from small sets to generate 50,000 positive pairs from the 10 large and 5 small heldout sets
	#multiply the number of positive pairs in the training data by 1, and generate that many negative paris per set
	#for each split, generate test data as:
	#use all data for held out small set, use all data for held out large set, multiply positive data by 1 and generate that much negative data
	#write train/test data to heldout50/
	genCrossHeldOutDataWithGroups(protGroups,protGroupInts,((2000,4000,3125,6250,1,'Train_'),[('all','all',1,'Test_')]),saveFolderName+'HeldOut50/')

	#create 21 group held-out validation using 50/50 splits
	#for each split, generate training data:
	#if held out data is small set, use 1600 pairs from large sets and 800 pairs from small sets to generate 50,000 positive pairs from the 10 large and 5 small heldout sets
	#if held out data is large set, use 2500 pairs from large sets and 1250 pairs from small sets to generate 50,000 positive pairs from the 10 large and 5 small heldout sets
	#multiply the number of positive pairs in the training data by 4, and generate that many negative paris per set
	#for each split, generate test data as:
	#split 1
	#use all data for held out small set, use all data for held out large set, multiply positive data by 9 and generate that much negative data
	#split 2
	#use all data for held out small set, use all data for held out large set, multiply positive data by 997/3 and generate that much negative data
	#write train/test data to heldout50/
	genCrossHeldOutDataWithGroups(protGroups,protGroupInts,((800,1600,1250,2500,4,'Train_'),[('all','all',9,'Test1_'),(300,600,(997/3),'Test2_')]),saveFolderName+'HeldOut20/')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parses BioPhysical interactions from BioGRID to create a final output file')
	parser.add_argument('--InteractionFileName',help='File to read Interactions from',default=currentDir+'HumanUniprotEntrezInteractionLst.tsv')
	parser.add_argument('--ProteinDataFileName',help='File to read proteins from',default=parentDir+'HumanUniprotEntrezProteinLst.tsv')
	parser.add_argument('--SaveFolderName',help='Folder to save new datasets into',default=currentDir)
	parser.add_argument('--ProteinDataType',help='Type of data for protein names, either int or string',default='int')
	parser.add_argument('--Seed',help='Random Seed to use',default=1,type=int)
	args = parser.parser_args()

	createDataSplits(args.InteractionFileName,args.ProteinDataFileName,args.SaveFolderName,args.ProteinDataType,args.Seed)