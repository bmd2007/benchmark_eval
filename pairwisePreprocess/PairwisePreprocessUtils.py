import numpy as np

#takes a torch matrix, and applies calculations to create single value aggregations
def matrixCalculations(mat,prefix='',dictionary={}):
	if mat is None:
		for item in ['sum','avg','avgmax','prod','max']:
			dictionary[prefix+item] = '?'
		return dictionary
		
	
	dictionary[prefix+'sum'] = mat.sum().item()
	dictionary[prefix+'avg'] = mat.mean().item()
	dictionary[prefix+'avgmax'] = ((mat.max(dim=0)[0].sum()/mat.shape[1] + mat.max(dim=1)[0].sum()/mat.shape[0])/2).item()
	dictionary[prefix+'prod'] = 1-((1-mat).prod()).item()
	dictionary[prefix+'max'] = mat.max().item()
	
#create a full matrix from a larger matrix given lists of rows and columns
def getMatFromMatrix(mat,coordsA,coordsB):
	if coordsA.shape[0] == 0 or coordsB.shape[0] == 0:
		return None
	xCoord = coordsA.repeat_interleave(coordsB.shape[0])
	yCoord = coordsB.repeat(coordsA.shape[0])
	newMat = mat[xCoord,yCoord]
	newMat = newMat.view(coordsA.shape[0],coordsB.shape[0])
	return newMat


#takes a numpy matrix, and applies calculations to create single value aggregations
#used for sparse matrices (which don't allow for indexing with tensors)
def matrixCalculationsNP(mat,prefix='',dictionary={}):
	if mat is None:
		for item in ['sum','avg','avgmax','prod','max']:
			dictionary[prefix+item] = '?'
		return dictionary
		
	dictionary[prefix+'sum'] = mat.sum()
	dictionary[prefix+'avg'] = mat.mean()
	dictionary[prefix+'avgmax'] = (mat.max(axis=0).sum()/mat.shape[1] + mat.max(axis=1).sum()/mat.shape[0])/2
	dictionary[prefix+'prod'] = 1-((1-mat).prod())
	dictionary[prefix+'max'] = mat.max()
	
#create a full matrix from a larger numpy matrix given lists of rows and columns
def getMatFromMatrixNP(mat,coordsA,coordsB):
	if coordsA.shape[0] == 0 or coordsB.shape[0] == 0:
		return None
	xCoord = np.repeat(coordsA,coordsB.shape[0])
	yCoord = np.tile(coordsB,coordsA.shape[0])
	newMat = mat[xCoord,yCoord]
	newMat = newMat.reshape(coordsA.shape[0],coordsB.shape[0])
	return newMat
		
		
#get the GO labels per protein
def getGOTerms(goTermLstFName,uniprotGeneMapping=None):
	goTerms = {}
	f = open(goTermLstFName)
	header = None
	for line in f:
		line = line.strip().split()
		if header is None:
			header ={}
			for i in range(0,len(line)):
				header[line[i]] = i
			continue
		uniName = line[header['UniprotName']]
		branch = line[header['Branch']]
		term = line[header['Term']]
		nameLst = [uniName]
		if uniprotGeneMapping is not None:
			if uniName not in uniprotGeneMapping:
				continue
			else:
				nameLst = uniprotGeneMapping[uniName]
		for uniName in nameLst:
			if uniName not in goTerms:
				goTerms[uniName] = {'BP':set(),'MF':set(),'CC':set()}
			goTerms[uniName][branch].add(term)
		
	f.close()
	return goTerms

#returns dictionary mapping each uniprot term to a list of entrez gene ids		
def createUniprotMapping(uniprotMappingFile):
	f = open(uniprotMappingFile)
	uniMap = {}
	header = None
	for line in f:
		line = line.strip().split('\t')
		if header is None:
			header = {}
			for i in range(0,len(line)):
				header[line[i]] = i
			continue
		uniprotID = line[header['Uniprot']]
		entrezID = line[header['Entrez']]
		if uniprotID not in uniMap:
			uniMap[uniprotID] = set()
		uniMap[uniprotID].add(entrezID)
	f.close()
	return uniMap
	

#given a list of pairs, an output file, and a list of scorers, score all protein pairs and output the results to the given file	
def scorePairs(pairs,fname,scorers,header=None):	
	f = open(fname,'w')
	for pair in pairs:
		scores = {}
		for scorer in scorers:
			scorer.scoreProteinPair(pair[0],pair[1],scores)
		if header is None:
			header = ['protein1','protein2']
			for item in scores:
				header.append(item)
			f.write('\t'.join(header)+'\n')
		
		scoreLst = [pair[0],pair[1]]
		for item in header[2:]:
			scoreLst.append(scores[item])
		f.write('\t'.join([str(s) for s in scoreLst])+'\n')
	f.close()


	
#given a lst if tuples, with the first part being the protein pair, and the second part being indices into a list of files
#along with scorers and a list of files, score all pairs, and output their results to the appropriate files marked by the per pair index list
def scoreAllPairs(pairToFileLst,scorers,allFiles):
	first = True
	header = ['protein1','protein2']
	for tup in pairToFileLst:
		pair, filesIdx = tup
		d = {}
		for sc in scorers:
			sc.scoreProteinPair(pair[0],pair[1],d)
		
		if first:
			for name in d:
				header.append(name)
			for f in allFiles:
				f.write('\t'.join(header)+'\n')
			first = False
		results = [pair[0],pair[1]]
		for item in header[2:]:
			results.append(d[item])
		for fIdx in filesIdx:
			f = allFiles[fIdx]
			f.write('\t'.join(str(s) for s in results)+'\n')
	
