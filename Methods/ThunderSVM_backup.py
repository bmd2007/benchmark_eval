import numpy as np
import thundersvm
#import libsvm.svmutil as libSVM
class ThunderSVM(object):
	def __init__(self,hyperParams):
		self.hyperParams = hyperParams
		self.svmModel = None
		#whether or not to flip the classes
		#this is a weird error, but, it seems like the svm doesn't remember the order of the classes
		#what this code currently does, in the fit section, is predict the training data,
		#compare where the positive interactions are in the those predictions compared to flipping the predictions
		#and determines if the order of the classes has been flipped or not, setting this variable to flip them
		#after each prediction if necessary.  If I can figure out why this is happening, I will remove the code and fix it.
		self.flipClasses = False
		
		
	def genModel(self):
		
		prob = True if 'b' not in self.hyperParams else (True if self.hyperParams['b'] not in [0,'0',False] else False)
		
		kernel='rbf' if not 'k' in self.hyperParams else self.hyperParams['k']
		degree=3 if not 'd' in self.hyperParams else self.hyperParams['d']
		gamma='auto' if not 'g' in self.hyperParams else self.hyperParams['g']
		coef0=0.0 if not 'c0' in self.hyperParams else self.hyperParams['c0']
		C=1.0 if not 'c' in self.hyperParams else self.hyperParams['c']
		tol=0.001 if not 't' in self.hyperParams else self.hyperParams['t']
		classWeight=None if not 'w' in self.hyperParams else self.hyperParams['w']
		maxIter = -1 if not 'i' in self.hyperParams else self.hyperParams['i']
		randomState = 1 if not 'r' in self.hyperParams else self.hyperParams['r']
		nJobs=-1 if not 'j' in self.hyperParams else self.hyperParams['j']
		
		self.svmModel = thundersvm.SVC(kernel,degree,gamma,coef0,C,tol,probability=prob,class_weight=classWeight,max_iter=maxIter,n_jobs=nJobs,random_state=randomState,gpu_id=0)
		
	
	def fit(self,features,classes):
	
		self.genModel()
		classes = classes.astype(np.int)
		self.svmModel.fit(features,classes)
		
		
		#weird class flipping check
		d = self.predict_proba(features)
		
		classData = classes[np.argsort(d[:,0])] #sort from most positive (least negative) to most negative
		z = np.arange(1,classData.shape[0]+1)
		correct = np.sum(classData/z) #score classes as they are
		z = np.arange(classData.shape[0],0,-1)
		incorrect = np.sum(classData/z) #score classes in reverse order
		
		print(correct,incorrect)
		
		if correct<incorrect:
			#Average Precions greater in reverse than in the current direction
			#(similar to AUC <0.5)
			#somehow, the classes got flipped
			self.flipClasses = True
		
		
		
		
	def predict_proba(self,features):
		lst = []
		#switch to a for loop to avoid a windows error generated when running thundersvm
		#the error is not memory related, as the data is using less than 10% of by GPU and CPU memory, but, I also
		#cannot debug  OSError: [WinError -529697949] Windows Error 0xe06d7363, and, for some reason, a for loop fixes it
		#may delete this later if I can figure out how to fix the error
		#error occurs in file "thundersvm.py", line 337, in _dense_predict
		for i in range(0,features.shape[0],1000):
			lst.append(self.svmModel.predict_proba(features[i:(i+1000)]))
		results = np.vstack(lst)
		if self.flipClasses:
			results = np.hstack((np.expand_dims(results[:,1],axis=1),np.expand_dims(results[:,0],axis=1)))
		return results



	def save(self,fname):
		self.svmModel.save_to_file(fname)
		
	def load(self,fname):
		self.genModel()
		self.svmModel.load_from_file(fname)
		