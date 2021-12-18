import numpy as np
import thundersvm
#import libsvm.svmutil as libSVM
class ThunderSVM(object):
	def __init__(self,hyperParams):
		self.hyperParams = hyperParams
		self.svmModel = None
		
		self.flipClasses=False
		
	def genModel(self):
		
		prob = True
		svmType = self.hyperParams.get('-s','SVC')
		svmType = svmType.upper()
		if svmType not in ['SVC','NUSVC','SVR','NUSVR']:
			print('error, unrecongized svm type',svmType)
			exit(42)
		
		if '-k' in self.hyperParams:
			self.hyperParams['-k']=self.hyperParams['-k'].lower()
			
		if svmType == 'SVC':

			self.svmModel = thundersvm.SVC(
				kernel=self.hyperParams.get('-k','rbf'),
				degree=self.hyperParams.get('-d',3),
				gamma=self.hyperParams.get('-g','auto'),
				coef0=self.hyperParams.get('-c0',0.0),
				C=self.hyperParams.get('-c',1.0),
				tol=self.hyperParams.get('-tol',0.001),
				probability = prob,
				class_weight=self.hyperParams.get('-w',None),
				shrinking=self.hyperParams.get('-shrink',False),
				cache_size=self.hyperParams.get('-cs',None),
				verbose=self.hyperParams.get('-verbose',False),
				max_iter=self.hyperParams.get('-iter',-1),
				n_jobs=self.hyperParams.get('-jobs',-1),
				max_mem_size=self.hyperParams.get('-max_mem',-1),
				random_state=self.hyperParams.get('seed',1),
				decision_function_shape=self.hyperParams.get('-df','ovo'),
				gpu_id=self.hyperParams.get('-gpu',0))
		
		elif svmType == 'NUSVC':
		
			self.svmModel = thundersvm.NuSVC(
				kernel=self.hyperParams.get('-k','rbf'),
				degree=self.hyperParams.get('-d',3),
				gamma=self.hyperParams.get('-g','auto'),
				coef0=self.hyperParams.get('-c0',0.0),
				nu=self.hyperParams.get('-nu',1.0),
				tol=self.hyperParams.get('-tol',0.001),
				probability = prob,
				shrinking=self.hyperParams.get('-shrink',False),
				cache_size=self.hyperParams.get('-cs',None),
				verbose=self.hyperParams.get('-verbose',False),
				max_iter=self.hyperParams.get('-iter',-1),
				n_jobs=self.hyperParams.get('-jobs',-1),
				max_mem_size=self.hyperParams.get('-max_mem',-1),
				random_state=self.hyperParams.get('seed',1),
				decision_function_shape=self.hyperParams.get('-df','ovo'),
				gpu_id=self.hyperParams.get('-gpu',0))
				
		elif svmType == 'SVR':
			self.svmModel = thundersvm.SVR(
				kernel=self.hyperParams.get('-k','rbf'),
				degree=self.hyperParams.get('-d',3),
				gamma=self.hyperParams.get('-g','auto'),
				coef0=self.hyperParams.get('-c0',0.0),
				C=self.hyperParams.get('-c',1.0),
				epsilon=self.hyperparams.get('-eps',0.1),
				tol=self.hyperParams.get('-tol',0.001),
				probability = prob,
				shrinking=self.hyperParams.get('-shrink',False),
				cache_size=self.hyperParams.get('-cs',None),
				verbose=self.hyperParams.get('-verbose',False),
				max_iter=self.hyperParams.get('-iter',-1),
				n_jobs=self.hyperParams.get('-jobs',-1),
				max_mem_size=self.hyperParams.get('-max_mem',-1),
				gpu_id=self.hyperParams.get('-gpu',0))
		
		elif svmType == 'NUSVR':
			self.svmModel = thundersvm.NuSVR(
				kernel=self.hyperParams.get('-k','rbf'),
				degree=self.hyperParams.get('-d',3),
				gamma=self.hyperParams.get('-g','auto'),
				coef0=self.hyperParams.get('-c0',0.0),
				nu=self.hyperParams.get('-nu',1.0),				
				C=self.hyperParams.get('-c',1.0),
				tol=self.hyperParams.get('-tol',0.001),
				probability = prob,
				shrinking=self.hyperParams.get('-shrink',False),
				cache_size=self.hyperParams.get('-cs',None),
				verbose=self.hyperParams.get('-verbose',False),
				max_iter=self.hyperParams.get('-iter',-1),
				n_jobs=self.hyperParams.get('-jobs',-1),
				max_mem_size=self.hyperParams.get('-max_mem',-1),
				gpu_id=self.hyperParams.get('-gpu',0))
				
	
	def fit(self,features,classes):
	
		self.genModel()
		classes = classes.astype(np.int)
		totalClasses = set(classes.tolist())
		#print(classes[0],totalClasses)
		
		classLst = sorted(set(classes.tolist()))
		#libsvm/thundersvm remember classes in the order seen, ensure the first few points are sorted in order
		for i in range(0,len(classLst)):
			a = np.argmax(classes==classLst[i])
			if a != i:#move first instance of class classLst[i] from slot a into slot id
				temp = classes[a]
				classes[a] = classes[i].copy()
				classes[i] = temp
				temp = features[a,:].copy()
				features[a,:] = features[i,:]
				features[i,:] = temp
		
		self.svmModel.fit(features,classes)
		
		posClassLabel = self.hyperParams.get('PosClassLabel',1)
		
#		#if positive was the first class seen, flip the results after predicting to ensure it comes back second
#		if posClassLabel is not None and posClassLabel in totalClasses and  classes[0] == posClassLabel:#postive was first class seen, so flip labels to make negative (0) first
#			self.flipClasses = True
#		
#		elif posClassLabel is None or posClassLabel not in totalClasses:
#			#we have no idea what the positive class label is
#			#but, the svm should score well on the training data, so we can figure it out by predicting the training data
#			
#			#class flipping check
#			d = self.predict_proba(features)
#			
#			classData = classes[np.argsort(d[:,0])] #sort from most positive (least negative) to most negative
#			z = np.arange(1,classData.shape[0]+1)
#			correct = np.sum(classData/z) #score classes as they are
#			z = np.arange(classData.shape[0],0,-1)
#			incorrect = np.sum(classData/z) #score classes in reverse order
#			
#			if correct<incorrect:
#				#Average Precions greater in reverse than in the current direction
#				#(similar to AUC <0.5)
#				#the classes are flipped
#				self.flipClasses = True
		
		
		
		
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
#		f = open(fname+'_flipped','w')
#		if self.flipClasses:
#			f.write('T')
#		else:
#			f.write('F')
#		f.close()
	def load(self,fname,flip=False):
		#print('l0')
		self.genModel()
		#print('l1',fname)
		self.svmModel.load_from_file(fname)
		#print('l2')
		try:
			f = open(fname+'_flipped')
			txt = f.read()
			f.close()
			if txt == 'T':
				self.flipClasses = True
			elif txt == 'F':
				self.flipClasses = False
		except:
			pass
		
		