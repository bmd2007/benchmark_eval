import argparse
import heapq
from goatools.base import get_godag

import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import PPIPUtils

import goatools.semantic as semantic
from goatools.associations import dnld_assc
from goatools.semantic import resnik_sim
import torch
import time

def getRoots(goTerms):
	roots = {'biological_process': None, 'cellular_component': None, 'molecular_function': None}
	for name, item in goTerms.items():
		if item.depth == 0:
			roots[item.namespace] = item


def getAllL2Nodes(goTerms):
	nodeLst = []
	for node in goTerms:
		nodeLst.append(goTerms[node])
	nodeLst.sort(key = lambda x: x.depth) #work from top down
	
	visited = set()
	
	#for each node, move through its parents
	for node in nodeLst:
		if len(node.parents) == 0:
			node.distanceFromRoot = set([0])
		else:
			node.distanceFromRoot = set()
		visited.add(node)
		node.l2Nodes = set()
		#for each parent
		for p in node.parents:
			if p not in visited:
				continue #can only happen for terms with no instances in gaf file (filtered out in our code, but not in goatools)
				
			#give this node a distance from the root +1 for each parent path
			for num in p.distanceFromRoot:
				node.distanceFromRoot.add(num+1)
			#union all of the parents nodes that were 2 from the root
			node.l2Nodes |= p.l2Nodes

		#if this node has a distance from root path of 2, add it to the l2Nodes set
		if 2 in node.distanceFromRoot:
			node.l2Nodes.add(node.id)

#Compute the MCA/MICA/ Maximal Information Content Ancestor
#By removing through all nodes ordered by information content (ascending)
#And assigning all pairwise combinations of their children to have the given node as its MCA
def getMCAMatrix(goTerms,nameMap,descendents,deviceType='cpu',rev=False):
	print(deviceType)
	mat = torch.zeros((len(goTerms),len(goTerms)),device=deviceType).long()-1
	nodeLst = []
	for node in goTerms:
		nodeLst.append(goTerms[node])
	nodeLst.sort(key = lambda x: (x.icVal if not rev else -x.icVal)) #work from top down if rev is false.  Otherwise, doing computations on descendents
	idx = 0
	for node in nodeLst:
		curIdx = nameMap[node.id]
		mat[curIdx,curIdx] = curIdx
		#get parents and self
		children = torch.where(descendents[curIdx,:]>0)[0]
		idx += 1
		
		#mark all pairs of this node's children as descendents of this node
		xCoord = children.repeat_interleave(children.shape[0])
		yCoord = children.repeat(children.shape[0])
		mat[xCoord,yCoord] = curIdx
	return mat

def geneOntologyL2Creator(goFile,gafFile,saveFolder,uniprotGeneMapping=None,deviceType='cpu'):
	totalT = time.time()
	if saveFolder[-1] != '/':
		saveFolder += '/'
		
	PPIPUtils.makeDir(saveFolder)
	
	goTerms = get_godag(goFile)
	getAllL2Nodes(goTerms) #adds list of level 2 terms to each node
	termMap = {}
	for name,node in goTerms.items():
		if name not in termMap:
			termMap[name] = set()
		for term in node.l2Nodes:
			termMap[name].add(term)
	
	
	allProts = {}
	for ns in ['BP','CC','MF']:
		mapping = dnld_assc(gafFile,goTerms,namespace=ns)
		protToL2 = {}
		for name,terms in mapping.items():
			nameLst = [name]
			if uniprotGeneMapping is not None:
				if name not in uniprotGeneMapping:
					continue
				nameLst = uniprotGeneMapping[name]
			for n in nameLst:
				if n not in protToL2:
					protToL2[n] = set()
				for term in terms:
					protToL2[n] |= goTerms[term].l2Nodes
					#protToL2[n] |= termMap[term]
					#protToL2[n].add(term)
		allProts[ns] = protToL2
	
	#write term list to file
	allTermLst = [['ProtName','Namespace','GO L2 Term']]
	for name,prots in allProts.items():
		for prot, termLst in prots.items():
			for term in termLst:
				allTermLst.append([prot,name,term])
	
	PPIPUtils.writeTSV2DLst(saveFolder+'L2Terms.tsv',allTermLst)
	
	#calculate intersections/unions per protein at the l2 level
	for ns in ['BP','CC','MF']:
		termLookup = {}
		protLookup = {}
		allProtsNames = sorted(allProts[ns].keys())
		for item in allProtsNames:
			protLookup[item]=len(protLookup)
		for prot,termLst in allProts[ns].items():
			for term in termLst:
				if term not in termLookup:
					termLookup[term] = len(termLookup)
		mat = torch.zeros((len(protLookup),len(termLookup)),device=deviceType)
		for prot,termLst in allProts[ns].items():
			for term in termLst:
				mat[protLookup[prot],termLookup[term]] = 1
				
		#compute intersection
		mat = mat @ mat.T
		mat = mat.tolist()
		PPIPUtils.writeTSV2DLstHalf(saveFolder+'L2Intersections_'+ns+'.tsv',mat,allProtsNames)
		
		#union data
		lst = [['Prot Name','Prot Idx','Prot # Terms']]
		for prot in allProtsNames:
			lst.append([prot,protLookup[prot],len(allProts[ns][prot])])
		
		PPIPUtils.writeTSV2DLst(saveFolder+'L2Unions_'+ns+'.tsv',lst)
	print(time.time()-totalT)
	
def getDistMatrix(goTerms,nameLookup,deviceType='cpu',highVal = 1000,rev=False):
	mat = torch.zeros((len(goTerms),len(goTerms)),device=deviceType)+highVal
	allTerms = []
	for name,item in goTerms.items():
		allTerms.append((item.depth,nameLookup[name],item))
	
	#sort from root to lowest depth if rev=False
	#otherwise, create inverted tree by building from leaves up 
	allTerms.sort(reverse=rev) 
	#rows contain parents, will transpose at end of function (so rows contain children)
	for (depth,idx,node) in allTerms:
		lst = []
		if (len(node.parents) > 0 and rev == False) or (len(node.children)>0 and rev==True):
			#get all parents row ids
			if not rev:
				for item in node.parents:
					lst.append(nameLookup[item.id])
			else:
				for item in node.children:
					lst.append(nameLookup[item.id])
			lst = torch.tensor(lst)
			#get the min distance for all parents to all other nodes, + 1
			mat[idx,:] = mat[lst,:].min(axis=0)[0]+1
		mat[idx,idx]= 0
		
			
	#reset the high value mark
	mat[mat>=highVal] = highVal
	return mat.T


def geneOntologySSCreator(goFile,gafFile,saveFolder,calcDescentScore=False,deviceType='cpu'):
	totalT = time.time()
	
	if saveFolder[-1] != '/' and saveFolder[-1] != '\\':
		saveFolder += '/'
		
	PPIPUtils.makeDir(saveFolder)
	
	#get tree and term counts
	
	#tree attributes (per node)
	#	self.id = ""                # GO:NNNNNNN
	#   self.name = ""              # description
	#   self.namespace = ""         # BP, CC, MF   (biological_process, molecular_function, and cellular_component)
	#   self._parents = set()       # is_a basestring of parents
	#   self.parents = []           # parent records
	#   self.children = []          # children records
	#   self.level = None           # shortest distance from root node
	#   self.depth = None           # longest distance from root node
	#   self.is_obsolete = False    # is_obsolete
	#   self.alt_ids = set()        # alternative identifiers	
	goTerms = get_godag(goFile)
	
	
	termsPerUniprot = {}
	
	termCounts = {}
	for item in [('BP','biological_process'),('CC','cellular_component'),('MF','molecular_function')]:
		mapping = dnld_assc(gafFile,goTerms,namespace=item[0])
		for prot,termLst in mapping.items():
			if prot not in termsPerUniprot:
				termsPerUniprot[prot] = {}
			termsPerUniprot[prot][item[0]] = set(termLst)
		termCounts[item[1]] = semantic.TermCounts(goTerms, mapping)
	
	lst = [['UniprotName','Branch','Term']]
	for prot,ontLst in termsPerUniprot.items():
		for ont,termLst in ontLst.items():
			for term in termLst:
				lst.append([prot,ont,term])
	
	PPIPUtils.writeTSV2DLst(saveFolder+'GOTermLst.tsv',lst)
	
	
	infoContent = {'biological_process':[],'cellular_component':[],'molecular_function':[]}
	nodeLookup = {}
	for name,node in goTerms.items():
		#exclude unused terms
		if termCounts[node.namespace].get_term_freq(name) > 0:
			nodeLookup[name] = len(infoContent[node.namespace])
			icVal = semantic.get_info_content(name,termCounts[node.namespace])
			infoContent[node.namespace].append(node)
			node.icVal = icVal
		
	goTermsFiltered = {}
	goTermsFilteredNamespace = {'biological_process':{},'cellular_component':{},'molecular_function':{}}
	for item in nodeLookup:
		goTermsFiltered[item] = goTerms[item]
		goTermsFilteredNamespace[goTerms[item].namespace][item] = goTerms[item]
	for item in goTermsFilteredNamespace:
		print(item,len(goTermsFilteredNamespace[item]))
			
	
	#filter out children with no semantec value.  Used if descendent calculations are done.
	for name,node in goTermsFiltered.items():
		children = set()
		for item in node.children:
			if item.id in goTermsFiltered:
				children.add(item)
		node.children = children
	
	
	for item in infoContent:
		infoContent[item].sort(key=lambda x: x.icVal)
	
	#value larger than the distance between any two connect GO terms, used in creating distance matrix
	highVal =1000
	
	#list containing lookup info for all nodes, used later to write to output file
	lookupLst = [['Namespace','GO Name','LookupIDX','IC Val','Max Child IC Val','Depth']]
	
	
	for ns in infoContent:
		#create a distance matrix by traversing the goTerm tree
		distMat = getDistMatrix(goTermsFilteredNamespace[ns],nodeLookup,deviceType,highVal)
		
		#get descendents matrix
		descendents = torch.zeros(distMat.shape,device=deviceType).long()
		descendents[distMat<highVal] = 1 #make children/descendents 1
		descendents2 = descendents.to('cpu').tolist()
		PPIPUtils.writeTSV2DLst(saveFolder+'Descendents_'+ns+'.tsv',descendents2)

		#create a matrix containing the lookup id of each maximal common ancestor
		mcaMat = getMCAMatrix(goTermsFilteredNamespace[ns],nodeLookup,descendents,deviceType)
		
		#create a vector containing the icVal at each idx used in the matrix
		infoContentLookup = torch.zeros(len(goTermsFilteredNamespace[ns]),device=deviceType)
		for name,node in goTermsFilteredNamespace[ns].items():
			infoContentLookup[nodeLookup[name]] = node.icVal
		
		#create a resnik information content matrix
		icMat = infoContentLookup[mcaMat]
		
		
		#compute wu similarity, Improving the Measurement of Semantic Similarity between Gene Ontology Terms and Gene Products: Insights from an Edge- and IC-Based Hybrid Method
		#to compute y, we need to compute the distance from distance matrix, for each row of mcaMat and the row index
		#and compute the distance from the distance matrix for each column of mcaMat and the col index
		#and average the two together.  This can be done with torch.gather
		wu = torch.gather(distMat,1,mcaMat) + torch.gather(distMat,1,mcaMat.T)
		
		#wu formula = 1/(1+y) * a/(a+b)
		#1/(1+y)
		wu = 1/(1+wu)
		#a is resnik/icMat, so we have that, just need beta
		#beta is the average of both terms most informative child minus the sum of both terms icval
		#find most informative children
		
		descendents = descendents * infoContentLookup #make all children/descendents = icVal
		maxDescendents = descendents.max(axis=1)[0] #get max ic content over all descendents
		#calc difference between IC val and max Descendent ID val
		maxDescendents = maxDescendents - infoContentLookup
	
		#set each pair [a,b] = distMat[a] + distMat[b]
		descendents = (maxDescendents + maxDescendents.unsqueeze(0).T)/2
		#calculate a/(a+b)
		descendents = icMat/(icMat+descendents)
		descendents = torch.nan_to_num(descendents)
		#multiply wu (1/(1+y)) with distMat (a/(a+b))
		wu = wu * descendents
		
		
		#write per go term wu values to disk
		
		wu = wu.to('cpu').tolist()
		PPIPUtils.writeTSV2DLst(saveFolder+'Wu_'+ns+'.tsv',wu)
		
		
		
		
		#write resnik data to file
		icMatLst = icMat.to('cpu').tolist()
		PPIPUtils.writeTSV2DLst(saveFolder+'Resnik_'+ns+'.tsv',icMatLst)

#		Resnik Test
#		goKeys = list(goTermsFilteredNamespace[ns].keys())
#		for i in range(0,50):
#			idx1 = nodeLookup[goKeys[i]]
#			idx2 = nodeLookup[goKeys[i+1]]
#			print(goKeys[i],goKeys[i+1],idx1,idx2,resnik_sim(goKeys[i],goKeys[i+1],goTerms,termCounts[ns]),icMat[idx1][idx2])




		if calcDescentScore:

			#descendent calculations, calculate the value of the descendent SS by inverting the tree
			#create a distance matrix with leaves as the root, and the root ontology term as the common descendent
			distMatDesc = getDistMatrix(goTermsFilteredNamespace[ns],nodeLookup,deviceType,highVal,rev=True)
		
			#get descendents matrix or reversed distance matrix
			descendentsDesc = torch.zeros(distMatDesc.shape,device=deviceType).long()
			descendentsDesc[distMatDesc<highVal] = 1 #make children/descendents (technically ancestors) 1
			
			ancestors2 = descendentsDesc.to('cpu').tolist()
			PPIPUtils.writeTSV2DLst(saveFolder+'Ancestors_'+ns+'.tsv',ancestors2)

			
			#create a matrix containing the lookup id of each maximal common descendent
			mcaMatDesc = getMCAMatrix(goTermsFilteredNamespace[ns],nodeLookup,descendentsDesc,deviceType,rev=True)
			
			#create a resnik information content matrix
			#value to map to when no common descendent.  Setting high here, so it will be negative after next step, and filtered out into a zero aftewards
			infoContentLookup2 = torch.cat((infoContentLookup,torch.tensor([999],device=deviceType))) 
			
			mcaMatDesc[mcaMatDesc<0] = infoContentLookup2.shape[0]-1
			icMatDesc = infoContentLookup2[mcaMatDesc]
			
			#convert IC val to DSimresnik, = to ic_t1 + ic_t2 - ic_descend
			dSimResnik = infoContentLookup + (infoContentLookup.unsqueeze(0).T) - icMatDesc
			dSimResnik[dSimResnik < -500] = 0 #set to 0 when no common descendent
			#if ic_t1 + ic_t2 - ic_descend < 0, set to 1/ic_descend?
			dSimResnik[dSimResnik<0] = 1/icMatDesc[dSimResnik<0]
			
			#compute wu similarity, 
			#to compute y, we need to compute the distance from distance matrix, for each row of mcaMat and the row index
			#and compute the distance from the distance matrix for each column of mcaMat and the col index
			#and average the two together.  This can be done with torch.gather.
			#The number will be nonsense when no common descendent, but we muliply it by the dSimResnik matrix, which is set to 0 for those terms
			mcaMatDesc[mcaMatDesc==infoContentLookup2.shape[0]-1] = 0 #allowing pairs with no common descendents values to use GO Term 0, will multiply by zero later
			wuDesc = torch.gather(distMatDesc,1,mcaMatDesc) + torch.gather(distMatDesc,1,mcaMatDesc.T)
			#wu formula = 1/(1+y) * a/(a+b)
			#1/(1+y)
			wuDesc = 1/(1+wuDesc)
			ancestors = (infoContentLookup + infoContentLookup.unsqueeze(0).T)/2
		
			wuDesc = wuDesc * dSimResnik/(dSimResnik+ancestors)
				
			wuDesc = torch.nan_to_num(wuDesc)
					
			#write per go term wu values to disk
			
			wuDesc = wuDesc.to('cpu').tolist()
			PPIPUtils.writeTSV2DLst(saveFolder+'Desc_Wu_'+ns+'.tsv',wuDesc)
			
						
			#write resnik data to file
			dSimResnik = dSimResnik.to('cpu').tolist()
			PPIPUtils.writeTSV2DLst(saveFolder+'Desc_Resnik_'+ns+'.tsv',dSimResnik)
			

		for node in infoContent[ns]:
			idx = nodeLookup[node.id]
			lookupLst.append([ns,node.id,idx,node.icVal,maxDescendents[idx].item(),node.depth])

	
	
	PPIPUtils.writeTSV2DLst(saveFolder+'SS_Lookup.tsv',lookupLst)
	print('total',time.time()-totalT)
	
	exit(42)
			
	
