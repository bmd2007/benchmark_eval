import random
import os
import time
import threading
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import gzip
import os.path
import shutil
import urllib.request as request
import requests
import zipfile
import io

currentdir = os.path.dirname(os.path.realpath(__file__))



def getChromosomeNum(string):
	if string == 'X':
		return 23
	elif string == 'Y':
		return 24
	else:
		try:
			return int(string)
		except:
			return None

def parseConfigFile(configFile):
	config = open(configFile)
	settings = {}
	for line in config:
		if line[0] == '#':
			continue
		line = line.strip()
		if len(line) == 0:
			continue
		line = line.split('\t')
		settings[line[0]] = line[1]
	return settings

def doWork(st, sema,shell):
	try:
		print (st)
		t = time.time()
		lst = st.split()
		pipe = None
		if lst[-2] == '>':
			pipe = lst[-1]
			lst = lst[:-2]
		if shell:
			lst = ' '.join(lst)
		if not pipe:
			p = subprocess.Popen(lst,shell=shell)
			print(p.communicate())
		else:
			with open(pipe,'w') as output:
				p = subprocess.Popen(lst,stdout=output,shell=shell)
				print (p.communicate())
		print ('Time = ' + str(time.time()-t))
	except Exception as ex:
		print(ex)
		pass
	sema.release()

def runLsts(lsts,threads,shell=False):
	print('start')
	for i in range(0,len(lsts)):
		lst = lsts[i]
		numThreads = threads[i]
		sema = threading.Semaphore(numThreads) #max number of threads at once
		for item in lst:
			sema.acquire()
			t = threading.Thread(target=doWork, args=(item,sema,shell))
			t.start()
	
		for i in range(0,numThreads):
			sema.acquire()
		for i in range(0,numThreads):
			sema.release()
			

def calcPrecisionRecallLsts(lst):
	#finalR = []
	#finalP = []
	#finalR.append([])
	#finalP.append([])
	lst = np.asarray(lst)
	ind = np.argsort(-lst[:,0])
	lst = lst[ind,:]
	
	#get total true and cumulative sum of true
	totalPositive = np.sum(lst[:,1])
	totalNegative = lst.shape[0]-totalPositive
	
	finalR = np.cumsum(lst[:,1])
	FP = np.arange(1,finalR.shape[0]+1)-finalR
	
	
	#create precision array (recall counts / total counts)
	finalP = finalR/np.arange(1,lst.shape[0]+1)
	
	
	
	#find ties
	x = np.arange(finalR.shape[0]-1)
	ties = list(np.where(lst[x,0] == lst[x+1,0])[0])
	for idx in range(len(ties)-1,-1,-1):
		finalR[ties[idx]] = finalR[ties[idx]+1]
		finalP[ties[idx]] = finalP[ties[idx]+1]
		FP[ties[idx]] = FP[ties[idx]+1]

	TN = totalNegative - FP
	ACC = (TN + finalR)/finalR.shape[0]
	TNR = TN/totalNegative
	
	#scale recall from 0 to 1
	finalR = finalR / totalPositive	
	
	return (finalP,finalR,ACC,TNR)

def calcAndPlotCurvesLsts(predictionsLst,classLst,datanames,fileName,title,curveType,lineStyleLst=None,legFont=1,lineWidthLst=None,font=None,removeMargins=False,xMax=None,yMax=None,markerLst=None,colorLst=None,size=None,fig=None,dpi=300,reducePoints=None,frameon=True):
	
	xAxis = []
	yAxis = []
	for i in range(0,len(predictionsLst)):
		(prec,recall,acc,tnr) = calcPrecisionRecallLsts(np.hstack((np.expand_dims(predictionsLst[i],0).T,np.expand_dims(classLst[i],0).T)))
		if reducePoints:
			p = []
			r = []
			t = []
			totalPoints = len(prec)
			idxmov = (len(prec)//10)//(reducePoints//3) #1/3 of points in first 10%
			for i in range(0,len(prec)//10,idxmov):
				p.append(prec[i])
				r.append(recall[i])
				t.append(tnr[i])
			idxmov = (len(prec)//10*9)//(reducePoints//3*2) #2/3 of points in last 90%
			for i in range(len(prec)//10,len(prec),idxmov):
				p.append(prec[i])
				r.append(recall[i])
				t.append(tnr[i])
			p.append(prec[-1])
			r.append(recall[-1])
			t.append(tnr[-1])
			prec = np.asarray(p)
			recall = np.asarray(r)
			tnr=  np.asarray(t)
		if curveType == 'PRC':
			xAxis.append(recall)
			yAxis.append(prec)
		elif curveType == 'ROC':
			yAxis.append(recall)
			xAxis.append(1-tnr)
	plotLsts(xAxis,yAxis,datanames,fileName,title,curveType,lineStyleLst,legFont,lineWidthLst,font,removeMargins,xMax,yMax,markerLst,colorLst,size,fig,dpi,frameon=frameon)
	
	
def plotLstsSubplot(xAxis,yAxis,headerLst,fileName,title,fig, curveType='PRC',lineStyleLst=None,legFont=1,lineWidthLst=None,font=None,removeMargins=False,xMax=None,yMax=None,markerLst=None,colorLst=None,size=None,dpi=300,frameon=True):
	if fig is None:
		return
	xTickLabelSize = 8
	yTickLabelSize = 8
	#f.rc('figure', titlesize=32)  # fontsize of the figure title
	titleFontSize = 10
	#f.rc('axes', titlesize=16)     # fontsize of the axes title
	#f.rc('axes', labelsize=16)    # fontsize of the x and y labels
	axisLabelSize = 10
	#axisTitleSize = 16
	#f.rc('legend', fontsize=int(12*legFont))    # legend fontsize
	legFontSize=  int(8*legFont)

	if removeMargins:
		fig.margins(x=0)	
	if curveType == 'PRC':
		fig.set_xlabel('Recall',fontsize=axisLabelSize)
		fig.set_ylabel('Precision',fontsize=axisLabelSize)
	else:
		fig.set_xlabel('FPR',fontsize=axisLabelSize)
		fig.set_ylabel('Recall',fontsize=axisLabelSize)
	fig.set_title(title,fontsize=titleFontSize)
	for i in range(0,len(xAxis)):
		fig.plot(xAxis[i],yAxis[i],label=(None if headerLst is None else headerLst[i]),linestyle = ('solid' if lineStyleLst is None else lineStyleLst[i]),linewidth = (1 if lineWidthLst is None else lineWidthLst[i]),marker = (" " if markerLst is None else markerLst[i]),color = (None if colorLst is None else colorLst[i]))
	if headerLst is not None:
		leg = fig.legend(fontsize=legFontSize,frameon=frameon)
		if font is not None:
			plt.setp(leg.texts,family = font['family'])
	if yMax is not None:
		fig.set_ylim((0,yMax))
	else:
		fig.set_ylim((0,1.05))
	if xMax is not None:
		fig.set_xlim((0,xMax))
		
	for item in fig.get_xticklabels():
		item.set_family(font['family'])
		item.set_fontsize(xTickLabelSize)
	for item in fig.get_yticklabels():
		item.set_family(font['family'])
		item.set_fontsize(yTickLabelSize)
	




def plotLsts(xAxis,yAxis,headerLst,fileName,title,curveType='PRC',lineStyleLst=None,legFont=1,lineWidthLst=None,font=None,removeMargins=False,xMax=None,yMax=None,markerLst=None,colorLst=None,size=None,fig=None,dpi=300,frameon=True):
	if fig:
		plotLstsSubplot(xAxis,yAxis,headerLst,fileName,title,fig, curveType,lineStyleLst,legFont,lineWidthLst,font,removeMargins,xMax,yMax,markerLst,colorLst,size,dpi)
		return
		
	plt.cla()
	plt.clf()
		
	if font:
		plt.rc('font', **font)

	plt.rc('font', size=14)          # controls default text sizes
	plt.rc('axes', titlesize=16)     # fontsize of the axes title
	plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
	plt.rc('legend', fontsize=int(12*legFont))    # legend fontsize
	plt.rc('figure', titlesize=32)  # fontsize of the figure title
	if removeMargins:
		plt.margins(x=0)	
	if curveType == 'PRC':
		plt.xlabel('Recall')
		plt.ylabel('Precision')
	else:
		plt.xlabel('FPR')
		plt.ylabel('Recall')
	plt.title(title)
	for i in range(0,len(xAxis)):
		plt.plot(xAxis[i],yAxis[i],label=(None if headerLst is None else headerLst[i]),linestyle = ('solid' if lineStyleLst is None else lineStyleLst[i]),linewidth = (1 if lineWidthLst is None else lineWidthLst[i]),marker = (" " if markerLst is None else markerLst[i]),color = (None if colorLst is None else colorLst[i]))
	if headerLst is not None:
		plt.legend(frameon=frameon)
	if yMax is not None:
		plt.ylim((0,yMax))
	else:
		plt.ylim((0,1.05))
	if xMax is not None:
		plt.xlim((0,xMax))
	plt.tight_layout()
	if fileName is not None:
		if size is not None:
			plt.gcf().set_size_inches(size[0],size[-1])
		plt.savefig(fileName,dpi=300)

def calcAUPRCLsts(finalP,finalR,maxRecall=0.2):
	scores = []
	for i in range(0,len(finalR)):
		#do cutoffs
		cutoff = np.argmax(finalR[i]>maxRecall)
		if cutoff == 0:
			if maxRecall >= np.max(finalR[i]): #100% recall, grab all data
				cutoff = finalR[i].shape[0]-1
				
		cutoff+=1		
		pData = finalP[i][0:cutoff]
		rData = finalR[i][0:cutoff]
		
		
		
		#create binary vector of true values
		rData2 = np.hstack((np.zeros(1),rData[:-1]))
		rData = rData-rData2
		
		#calculate average precision at each true value
		x1 = np.sum(rData * pData)/np.sum(rData)
		
		scores.append(x1)
	return scores


def calcAUROC(classData,preds):
	fpr, tpr, thresholds = metrics.roc_curve(classData, preds)
	return metrics.auc(fpr, tpr)


def calcScores(classData,preds,thresholds):
	predictions = np.vstack((preds,classData)).T
	(prec,recall,acc,tnr) = calcPrecisionRecallLsts(predictions)
	scores = {'Avg Precision':[],'Max Precision':[],'Thresholds':thresholds,'AUC':0,'Acc':0,'Prec':0,'Recall':0}
	for item in thresholds:
		auprc = calcAUPRCLsts([prec],[recall],maxRecall=item)[0]
		precision = np.max(prec[recall>=item])
		scores['Avg Precision'].append(auprc)
		scores['Max Precision'].append(precision)

	auc = calcAUROC(classData,preds)
	maxAcc = np.max(acc)
	maxAccIdx = np.argmax(acc>=maxAcc)
	scores['AUC'] = auc
	scores['ACC'] = maxAcc
	scores['Prec'] = prec[maxAccIdx]
	scores['Recall'] = recall[maxAccIdx]
	return scores




def parseTSV(fname,form='string',delim='\t'):
	lst = []
	f = open(fname)
	for line in f:
		lst.append(line.strip().split(delim))
		if form == 'int':
			lst[-1] = [int(s) for s in lst[-1]]
		if form == 'float':
			lst[-1] = [float(s) for s in lst[-1]]
		lst[-1] = tuple(lst[-1])
	f.close()
	return lst

def parseTSVLst(fname,form=[],delim='\t'):
	lst = []
	f = open(fname)
	for line in f:
		line = line.strip().split(delim)
		for i in range(0,len(form)):
			if form[i] == 'int':
				line[i] = int(line[i])
			elif form[i] == 'float':
				line[i] = float(line[i])
		lst.append(tuple(line))
	f.close()
	return lst
	
def writeTSV2DLst(fname,lst,delim='\t'):
	f = open(fname,'w')
	for item in lst:
		item = delim.join([str(s) for s in item])
		f.write(item+'\n')
	f.close()

#takes a symmetric matrix, and writes the top right (N^2+N)/2 entries
#names fills in the first row and first column, optionally
def writeTSV2DLstHalf(fname,lst,names=None,delim='\t'):
	f = open(fname,'w')
	idx = 0
	if names is not None:
		f.write(delim+delim.join(names))
	for item in lst:
		item = delim.join([str(s) for s in item[idx:]])
		if names is not None:
			item = str(names[idx])+delim+item
		f.write(item+'\n')
		idx += 1
	f.close()
	
def createKFoldsAllData(data,k,seed=1,balanced=False):
	data = np.asarray(data)
	classData = np.asarray(data[:,2],dtype=np.int32)
	posData = data[classData==1,:]
	negData = data[classData==0,:]
	return createKFolds(posData.tolist(),negData.tolist(),k,seed,balanced)
	
def createKFolds(pos,neg,k,seed=1,balanced=False):
	random.seed(seed)
	np.random.seed(seed)
	if len(pos[0]) == 2:
		fullPos = []
		for item in pos:
			fullPos.append((item[0],item[1],1))
		fullNeg = []
		for item in neg:
			fullNeg.append((item[0],item[1],0))
		pos = fullPos
		neg = fullNeg
	if balanced:
		pos = np.asarray(pos)
		neg = np.asarray(neg)
		posIdx = [x for x in range(0,len(pos))]
		negIdx = [x for x in range(0,len(neg))]
		random.shuffle(posIdx)
		random.shuffle(negIdx)
		numEntries = min(pos.shape[0],neg.shape[0])
		pos = pos[posIdx[:numEntries],:]
		neg = neg[negIdx[:numEntries],:]
		pos = pos.tolist()
		neg = neg.tolist()
			
				
	posIdx = [x for x in range(0,len(pos))]
	negIdx = [x for x in range(0,len(neg))]
	random.shuffle(posIdx)
	random.shuffle(negIdx)
	trainSplits = []
	testSplits = []
	pos = np.asarray(pos)
	neg = np.asarray(neg)
	
	for i in range(0,k):
		startP = int((i/k)*pos.shape[0])
		endP = int(((i+1)/k)* pos.shape[0])
		startN = int((i/k)*neg.shape[0])
		endN = int(((i+1)/k)* neg.shape[0])
		if i == k-1:
			endP = pos.shape[0]
			endN = neg.shape[0]
		a = pos[posIdx[:startP],:]
		b = pos[posIdx[endP:],:]
		c = neg[negIdx[:startN],:]
		d = neg[negIdx[endN:],:]
		lst = np.vstack((a,b,c,d))
		np.random.shuffle(lst)
		trainSplits.append(lst)
		
		e = pos[posIdx[startP:endP],:]
		f = neg[negIdx[startN:endN],:]
		lst = np.vstack((e,f))
		np.random.shuffle(lst)
		testSplits.append(lst)
	return trainSplits, testSplits


	
	
def makeDir(directory):
	if directory[-1] != '/' and directory[-1] != '\\':
		directory += '/'
	try:
		if os.path.isdir(directory):
			pass
		else:
			os.mkdir(directory)
	except:
		pass

def formatScores(results,title):
	lst = []
	lst.append([title])
	lst.append(('Acc',results['ACC'],'AUC',results['AUC'],'Prec',results['Prec'],'Recall',results['Recall']))
	lst.append(('Thresholds',results['Thresholds']))
	lst.append(('Precision',results['Max Precision']))
	lst.append(('Avg Precision',results['Avg Precision']))
	return lst
	

def parseUniprotFasta(fileLocation, desiredProteins):
	f= gzip.open(fileLocation,'rb')
	curUniID = ''
	curAASeq = ''
	seqDict ={}
	desiredProteins = set(desiredProteins)
	for line in f:
		line = line.strip().decode('utf-8')
		if line[0] == '>':
			if curUniID in desiredProteins:
				seqDict[curUniID] = curAASeq
			line = line.split('|')
			curUniID = line[1]
			curAASeq = ''
		else:
			curAASeq += line
	f.close()
	if curUniID in desiredProteins:
		seqDict[curUniID] = curAASeq
	return seqDict
	

def downloadWGet(downloadLocation,fileLocation):
	runLsts([['wget '+downloadLocation + ' -O ' + fileLocation]],[1],shell=True)

def downloadFile(downloadLocation, fileLocation):
	data = request.urlopen(downloadLocation)
	f = open(fileLocation,'wb')
	shutil.copyfileobj(data,f)
	f.close()


def unZip(fileLocation,newFileLocation):
	z = zipfile.ZipFile(fileLocation)
	z.extractall(newFileLocation)


def downloadZipFile(downloadLocation, fileLocation):
	r = requests.get(downloadLocation)
	z = zipfile.ZipFile(io.BytesIO(r.content))
	z.extractall(fileLocation)
	
def getUniprotFastaLocations(getBoth=True):
	uniprotFastaLoc = currentdir+'/PPI_Datasets/uniprot_sprot.fasta.gz'
	uniprotFastaLoc2 = currentdir+'/PPI_Datasets/uniprot_trembl.fasta.gz'
	if not os.path.exists(uniprotFastaLoc):
		downloadFile('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz',uniprotFastaLoc)
	if not os.path.exists(uniprotFastaLoc2) and getBoth:
		downloadFile('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz',uniprotFastaLoc2)
	return uniprotFastaLoc, uniprotFastaLoc2
