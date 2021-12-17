import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import PPIPUtils
import torch
import PairwisePreprocessUtils
import time


class GeneOntologySemScorer(object):
	def __init__(self,goSaveFolder,descendents=False,uniprotGeneMapping=None,loadNamespaces = ['BP','CC','MF'],deviceType='cpu'):
		
		t = time.time()
		if goSaveFolder[-1] != '/' and goSaveFolder[-1] != '\\':
			goSaveFolder+='/'
		self.deviceType = deviceType
		goTerms = PairwisePreprocessUtils.getGOTerms(goSaveFolder+'GoTermLst.tsv',uniprotGeneMapping)
		
		#whether to compute descendent SS values
		self.descendents = descendents
		
		#convert goTerms to goIDXs in matrices
		#get terms and data
		data = PPIPUtils.parseTSV(goSaveFolder+'SS_Lookup.tsv')
		
		#map lookupIDX -> data and term-> lookupIdx
		header = {}
		for i in range(0,len(data[0])):
			header[data[0][i]] = i
		
		self.termToIdx = {}
		icData = {'BP':[],'CC':[],'MF':[]}
		nsLookup = {'biological_process':'BP','cellular_component':'CC','molecular_function':'MF'}
		for item in data[1:]:
			term = item[header['GO Name']]
			lookup = int(item[header['LookupIDX']])
			icVal = float(item[header['IC Val']])
			maxChildIC = float(item[header['Max Child IC Val']])
			ns = item[header['Namespace']]
			ns = nsLookup.get(ns,ns)
			icData[ns].append((lookup,icVal))
			self.termToIdx[term] = lookup
		print(len(self.termToIdx))
		for item in icData:
			m = 0
			for item2 in icData[item]:
				m = max(m,item2[0])
			print(item,m)
		self.icData = {'BP':[],'CC':[],'MF':[]}
		for item in self.icData:
			self.icData[item] = torch.zeros(len(icData[item]),device=self.deviceType)
			for pair in icData[item]:
				self.icData[item][pair[0]] = pair[1]
		
		#convert terms to idxs
		self.goTermsIdx = {}
		for prot,onts in goTerms.items():
			self.goTermsIdx[prot] = {}
			for ont,terms in onts.items():
				self.goTermsIdx[prot][ont] = set()
				for term in terms:
					if term in self.termToIdx:
						self.goTermsIdx[prot][ont].add(self.termToIdx[term])
				self.goTermsIdx[prot][ont] = torch.tensor(list(self.goTermsIdx[prot][ont]),device=self.deviceType)
				
		print('mapped',time.time()-t)
		revNSMap = {'BP':'biological_process','CC':'cellular_component','MF':'molecular_function'}
		t = time.time()
		#load resnik matrices
		self.resnikData = {}
		self.resnikThreshs = {}
		for item in loadNamespaces:
			self.resnikData[item] = torch.tensor(PPIPUtils.parseTSV(goSaveFolder+'Resnik_'+revNSMap[item]+'.tsv','float'),device=self.deviceType)
			self.resnikThreshs[item] = self.resnikData[item].flatten().sort()[0][int(self.resnikData[item].flatten().shape[0]*.9)]
			print(item,self.resnikData[item].shape)
		print('Resnik',time.time()-t)
		t = time.time()
		
		#load wu matrices
		self.wuData = {}
		for item in loadNamespaces:
			self.wuData[item] = torch.tensor(PPIPUtils.parseTSV(goSaveFolder+'Wu_'+revNSMap[item]+'.tsv','float'),device=self.deviceType)
			print(item,self.wuData[item].shape)
		print('Wu',time.time()-t)
		t = time.time()
		
		if self.descendents:
			for item in loadNamespaces:
				self.resnikData['Desc_'+item] = torch.tensor(PPIPUtils.parseTSV(goSaveFolder+'Desc_Resnik_'+revNSMap[item]+'.tsv','float'),device=self.deviceType)
				self.resnikThreshs['Desc_'+item] = self.resnikData['Desc_'+item].flatten().sort()[0][int(self.resnikData['Desc_'+item].flatten().shape[0]*.9)]
				print(item,self.resnikData['Desc_'+item].shape)
			print('ResnikDesc',time.time()-t)

			t = time.time()
			for item in loadNamespaces:
				self.wuData['Desc_'+item] = torch.tensor(PPIPUtils.parseTSV(goSaveFolder+'Desc_Wu_'+revNSMap[item]+'.tsv','float'),device=self.deviceType)
				print(item,self.wuData['Desc_'+item].shape)
			print('WuDesc',time.time()-t)
			t = time.time()
		
		
	def scoreProteinPair(self,id1,id2,dictionary={},prefix='GOSS_'):
		t = time.time()
		emptySet = {}
		for ns in self.resnikData:
			emptySet[ns] = None
		#get goTermIdxs per ontology for each protein
		p1Idxs = self.goTermsIdx.get(id1,emptySet)
		p2Idxs = self.goTermsIdx.get(id2,emptySet)
		
		#do calculations per ontology
		for ns in self.resnikData:
			curNSprefix = prefix+ns+'_'
			realNS = ns.split('_')[-1]
			idxs1 = p1Idxs[realNS]
			idxs2 = p2Idxs[realNS]
			#no GO terms for one ontology for at least one protein
			if idxs1 is None or idxs2 is None or idxs1.shape[0] == 0 or idxs2.shape[0] == 0:
				for item in ['Resnik_','ResnikCoverage_','Lin_','Jiang_','Rev_','Wu_']:
					PairwisePreprocessUtils.matrixCalculations(None,curNSprefix+item,dictionary)
				continue
				
			
			#get values
			resnikMat = PairwisePreprocessUtils.getMatFromMatrix(self.resnikData[ns],idxs1,idxs2)
			wuMat = PairwisePreprocessUtils.getMatFromMatrix(self.wuData[ns],idxs1,idxs2)
			
			icVal1 = self.icData[realNS][idxs1]
			icVal2 = self.icData[realNS][idxs2]
			#icVal1+icVal2
			icSumMat = icVal1.unsqueeze(0).T + icVal2
			
			#resnik computation (resnik(a,b))
			PairwisePreprocessUtils.matrixCalculations(resnikMat,curNSprefix+'Resnik_',dictionary)
			
			#get resnik, filtering by value existance instead of raw values
			resCoverage = torch.zeros(resnikMat.shape,device=self.deviceType)
			resCoverage[resnikMat>self.resnikThreshs[ns]] = 1
			PairwisePreprocessUtils.matrixCalculations(resnikMat,curNSprefix+'ResnikCoverage_',dictionary)
		
			#lin computation ((2*resnik(a,b))/(icVal1+icVal2)
			linMat = torch.clone(resnikMat)
			linMat = 2*linMat / icSumMat
			PairwisePreprocessUtils.matrixCalculations(linMat,curNSprefix+'Lin_',dictionary)
			
			#Jiang/Corath computation, 1/(1+ic1+ic2-2*resnik)
			jiangMat = icSumMat -2*torch.clone(resnikMat)
			jiangMat = 1/(1+jiangMat)
			PairwisePreprocessUtils.matrixCalculations(jiangMat,curNSprefix+'Jiang_',dictionary)
			
			#Schlicker relevance similairty  = (2*resnik)/(icVal1+icVal2) *(1-prob)
			#=lin * (1-prob), where prob = exp(-Resnik), since Resnik is -log(prob)
			#= lin * (1- exp(-Resnik))
			revMat = torch.exp(-1 * resnikMat)
			revMat = linMat * (1-revMat)
			PairwisePreprocessUtils.matrixCalculations(revMat,curNSprefix+'Rev_',dictionary)
			
			#Wu similarity
			PairwisePreprocessUtils.matrixCalculations(wuMat,curNSprefix+'Wu_',dictionary)
		
		return dictionary
			
			