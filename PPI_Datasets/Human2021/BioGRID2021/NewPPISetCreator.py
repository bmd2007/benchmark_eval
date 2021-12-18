import os
import sys
#add parent and grandparent and great-grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
import PPIPUtils
import numpy as np
import random

newDir = 'NewDatasets/'
PPIPUtils.makeDir(newDir)
random.seed(0)

#contain list of proteins, and interactions, that were not in any BioGRID interaction lists for biophysical interactions prior to 2017
nonIntGenes2017 = PPIPUtils.parseTSV('NonIntGenes2017.tsv')
newInts = PPIPUtils.parseTSV('NewInt2017.tsv')

allGenes = PPIPUtils.parseTSV('HumanUniprotEntrezProteinLst.tsv')
allGenes = np.asarray(allGenes)[1:,0].tolist()
allInts = PPIPUtils.parseTSV('HumanUniprotEntrezInteractionLst.tsv')


nonInt2017Set = set()
for item in nonIntGenes2017:
	nonInt2017Set.add(item[0])

newIntsSet = set()
for item in newInts:
	newIntsSet.add((item[0],item[1]))
	newIntsSet.add((item[1],item[0]))

oldInts = []
for item in allInts:
	if (item[0],item[1]) not in newIntsSet:
		oldInts.append(item)

allIntsSet = set()
for item in allInts:
	allIntsSet.add((item[0],item[1]))
	allIntsSet.add((item[1],item[0]))


posSet = set()
negSet = set()
print('train')
random.shuffle(oldInts)
f = open(newDir+'NewTrainSet.tsv','w')
for i in range(0,20000):
	f.write(oldInts[i][0]+'\t'+oldInts[i][1]+'\t'+'1\n')
	posSet.add(tuple(oldInts[i]))
	
for i in range(0,80000):
	a = allGenes[random.randint(0,len(allGenes)-1)]
	b = allGenes[random.randint(0,len(allGenes)-1)]
	if a == b or (a,b) in allIntsSet or (a,b) in negSet:
		continue
	negSet.add((a,b))
	negSet.add((b,a))
	tup = ((a,b))
	if int(a) > int(b):
		tup = ((b,a))
	f.write(tup[0]+'\t'+tup[1]+'\t'+'0\n')
f.close()




print('easy')
f = open(newDir+'NewSetEasyTest.tsv','w')
random.shuffle(newInts)
for i in range(0,300):
	posSet.add(newInts[i])
	f.write(newInts[i][0]+'\t'+newInts[i][1]+'\t'+'1'+'\n')
	

i = 0
while i < 99700:
	a = allGenes[random.randint(0,len(allGenes)-1)]
	b = allGenes[random.randint(0,len(allGenes)-1)]
	if a == b or (a,b) in allIntsSet or (a,b) in negSet:
		continue
	negSet.add((a,b))
	negSet.add((b,a))
	tup = ((a,b))
	if int(a) > int(b):
		tup = ((b,a))
	f.write(tup[0]+'\t'+tup[1]+'\t'+'0\n')
	i+=1
f.close()



print('hard',len(nonInt2017Set))
f = open(newDir+'NewSetHardTest.tsv','w')
random.shuffle(newInts)
i = 0
count = 0
while count < 300:
	if newInts[i][0] not in nonInt2017Set or newInts[i][1] not in nonInt2017Set or newInts[i] in posSet:
		i += 1
		continue
	posSet.add(newInts[i])
	f.write(newInts[i][0]+'\t'+newInts[i][1]+'\t'+'1'+'\n')
	i += 1
	count += 1
	
i = 0
while i < 99700:
	a = allGenes[random.randint(0,len(allGenes)-1)]
	b = allGenes[random.randint(0,len(allGenes)-1)]
	if a == b or (a,b) in allIntsSet or (a,b) in negSet:
		continue
	negSet.add((a,b))
	negSet.add((b,a))
	tup = ((a,b))
	if int(a) > int(b):
		tup = ((b,a))
	f.write(tup[0]+'\t'+tup[1]+'\t'+'0\n')
	i += 1
f.close()


for idx in range(0,10):
	f=open(newDir+'AllInts_'+str(idx)+'.tsv','w')
	random.shuffle(allInts)
	i = 0
	count = 0
	while count < 300:
		if allInts[i] in posSet:
			i += 1
			continue
		f.write(allInts[i][0]+'\t'+allInts[i][1]+'\t'+'1'+'\n')
		posSet.add(allInts[i])
		i += 1
		count += 1
	i = 0
	while i < 99700:
		a = allGenes[random.randint(0,len(allGenes)-1)]
		b = allGenes[random.randint(0,len(allGenes)-1)]
		if a == b or (a,b) in allIntsSet or (a,b) in negSet:
			continue
		negSet.add((a,b))
		negSet.add((b,a))
		tup = ((a,b))
		if int(a) > int(b):
			tup = ((b,a))
		f.write(tup[0]+'\t'+tup[1]+'\t'+'0\n')
		i += 1
	f.close()


for idx in range(0,10):
	f=open(newDir+'OldInts_'+str(idx)+'.tsv','w')
	random.shuffle(oldInts)
	i = 0
	count = 0
	while count < 300:
		if oldInts[i] in posSet:
			i += 1
			continue
		f.write(oldInts[i][0]+'\t'+oldInts[i][1]+'\t'+'1'+'\n')
		posSet.add(oldInts[i])
		i += 1
		count += 1
	
	i = 0
	while i < 99700:
		a = allGenes[random.randint(0,len(allGenes)-1)]
		b = allGenes[random.randint(0,len(allGenes)-1)]
		if a == b or (a,b) in allIntsSet or (a,b) in negSet:
			continue
		negSet.add((a,b))
		negSet.add((b,a))
		tup = ((a,b))
		if int(a) > int(b):
			tup = ((b,a))
		f.write(tup[0]+'\t'+tup[1]+'\t'+'0\n')
		i += 1
	f.close()