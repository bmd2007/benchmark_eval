import sys
import os
from SortedMatrixParser import SortedMatrixParser

folder = os.path.abspath(__file__)
folder = '/'.join(folder.split('/')[0:-2]) + '/'
sys.path.append(folder)


class SortedMatrixParserIOU(SortedMatrixParser):
	def __init__(self,matrixFile,unionFile,name,delim = '\t',delim2='\t',headerLines=1):
		SortedMatrixParser.__init__(self,matrixFile,name,delim)
		self.unionData ={}
		f = open(unionFile)
		for line in f:
			line = line.strip().split(delim2)
			if headerLines >0:
				headerLines-=1
				continue
			self.unionData[line[0]] = int(float(line[-1]))
		f.close()
		for item in self.geneSet:
			if item not in self.unionData:
				print('Error, invalid union data.')
				exit(42)
		
	def scoreProteinPair(self,id1,id2,retSet=None):
		if retSet is None:
			retSet = {}
		retSet[self.name+'_Int'] = '?'
		retSet[self.name+'_Union'] = '?'
		retSet[self.name+'_IOU'] = '?'
		retSet[self.name+'_IOM'] = '?'
		
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
			intVal = int(float(self.curLine[idx2-idx1+1])) #+1 to skip past row header (gene id)
			unionVal = self.unionData[id1] + self.unionData[id2] - intVal
			retSet[self.name+'_Int'] = intVal
			retSet[self.name+'_Union'] = unionVal
			retSet[self.name+'_IOU'] = 0 if intVal == 0 else intVal/unionVal
			retSet[self.name+'_IOM'] = 0 if intVal == 0 else intVal/max(self.unionData[id1],self.unionData[id2])
		return retSet
		