import numpy as np
class ProteinFeaturesHolder(object):
	def __init__(self,featLst,convertToInt=True):
		#note, we assume that all files in feature list contain 1 row per protein, and are in the same order
		#may add check/sorting fixes later if needed
		
		self.convertToInt = convertToInt
		self.header = []
		self.flst = []
		self.data = []
		self.rowHeader = []
		self.rowLookup = {}
		
		for fIdx in range(0,len(featLst)):
			self.flst.append(open(featLst[fIdx]))
			if fIdx == 0:
				self.header = self.header + self.flst[-1].readline().strip().split()
			else:
				self.header = self.header + self.flst[-1].readline().strip().split()[1:]
		
		while True:
			curLine = self.flst[0].readline().strip().split()
			if len(curLine) == 0:
				break
			protName = curLine[0]
			try:
				if self.convertToInt:
					protName = int(protName)
			except:
				pass
			curLine = curLine[1:]
			curLine = [float(s) for s in curLine]
			self.data.append(curLine)
			self.rowHeader.append(protName)
			self.rowLookup[protName] = len(self.data)-1
			for fIdx in range(1,len(self.flst)):
				curLine = self.flst[fIdx].readline().strip().split()[1:]
				curLine = [float(s) for s in curLine]
				self.data[-1] += curLine
			self.data[-1] = np.asarray(self.data[-1])
		self.data = np.asarray(self.data)
		for item in self.flst:
			item.close()
		self.flst = None
		
		
	def genData(self,pairs):
		plst1 = []
		plst2 = []
		for item in pairs:
			try:
				if self.convertToInt:
					i0 = int(item[0])
			except:
				i0 = item[0]
			try:
				if self.convertToInt:
					i1 = int(item[1])
			except:
				i1 = item[1]
			plst1.append(self.rowLookup[i0])
			plst2.append(self.rowLookup[i1])
		z = np.hstack((self.data[plst1,:],self.data[plst2,:]))
		return z