import sys
import os

folder = os.path.abspath(__file__)
folder = '/'.join(folder.split('/')[0:-2]) + '/'
sys.path.append(folder)

class SortedMatrixParser(object):
	def __init__(self,matrixFile,name,delim = '\t'):

		self.matrixFile = matrixFile
		self.delim = delim
		self.name =name
		self.openFile()
		
	def openFile(self):
		self.f = open(self.matrixFile)
		self.geneSet = self.f.readline().strip().split(self.delim)
		self.geneIdxs = {}
		for i in range(0,len(self.geneSet)):
			self.geneSet[i] = self.geneSet[i]
			self.geneIdxs[self.geneSet[i]] = i
		self.geneSet = set(self.geneSet)
		self.curLine = self.f.readline().strip().split(self.delim)
		self.curIdx = 0
	
	def readLine(self,newIdx):
		if newIdx < self.curIdx:
			print('Warning, points out of order.  Reopening file to find data')
			self.openFile()
		if self.curIdx != newIdx:
			while self.curIdx != newIdx:
				self.curLine = self.f.readline()
				self.curIdx += 1
			self.curLine= self.curLine.strip().split(self.delim)
		
	def scoreProteinPair(self,id1,id2,retSet=None):
		if retSet is None:
			retSet = {}
		retSet[self.name] = '?'
		if id1 > id2:
			t = id2
			id2 = id1
			id1 = t
			
		if id1 in self.geneSet and id2 in self.geneSet:
			idx1 = self.geneIdxs[id1]
			self.readLine(idx1)
			if str(id1) != self.curLine[0]:
				print('File format error.')
				exit(42)
			idx2 = self.geneIdxs[id2]
			retSet[self.name] = self.curLine[idx2-idx1+1] #+1 to skip past row header (gene id)
		return retSet
		