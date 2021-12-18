from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump, load

try:
	from LibSVM import LibSVM
except:
	pass
try:
	from ThunderSVM import ThunderSVM
except:
	pass


#Supports SVC's for ThunderSVM, LibSVM, and Sklearn

class GenericSVM(object):
	def __init__(self,hyperParams):
		SVM_TYPE_LOOKUPS={'SVC':0,'NUSVC':1,'EPSILON_SVR':3,'NUSVR':4}
		KERNEL_TYPE_LOOKUPS={'LINEAR':0,'POLYNOMIAL':1,'RBF':2,'SIGMOID':3,'PRECOMPUTED':4}
		
		SVM_TYPE_LIST = ['SVC', 'NUSVC', 'ONECLASS', 'SVR', 'NUSVR']
		KERNEL_TYPE_LIST = ['LINEAR', 'POLYNOMIAL', 'RBF', 'SIGMOID', 'PRECOMPUTED']
		
        #create a copy of the hyperparams
		self.hyperParams = hyperParams.copy()

		lst = [ ('-g','gamma'),
				('-k','kernel','-t'),
				('-s','svm_type','svmType'),
				('-c0','-r'),
				('-c','C'),
				('-nu','-n'),
				('-eps','-p','epsilon'),
				('-tol','-e','tol'),
				('-cs','-m','cache_size'),
				('-shrink','-h'),
				('-w','-wi','class_weight'),
				('-verbose','-q'),
				('-iter','max_iter'),
				('-verbose','verbose'),
				('-jobs','n_jobs'),
				('seed','random_state'),
				('lr','learning_rate'),
				('-d','degree'),
				('-df','decision_function_shape')
			]
				
		#handle multiple aliases for svm data
		for tup in lst:
			for i in range(0,len(tup)):
				if tup[i] in self.hyperParams:
					for j in range(0,len(tup)):
						self.hyperParams[tup[j]] = self.hyperParams[tup[i]]
					break
				
		modelType = self.hyperParams.get('Model','SVC').upper()
		
		#lib svm needs svm type and kernel type as numbers instead of strings
		if modelType != 'LIBSVM':
			if '-k' in self.hyperParams and type(self.hyperParams['-k']) is int:
				self.hyperParams['-k'] = KERNEL_TYPE_LIST[self.hyperParams['-k']]
			if '-s' in self.hyperParams and type(self.hyperParams['-s']) is int:
				self.hyperParams['-s'] = SVM_TYPE_LIST[self.hyperParams['-s']]
		else:
			if '-k' in self.hyperParams and type(self.hyperParams['-k']) is str:
				self.hyperParams['-k'] = KERNEL_TYPE_LOOKUP[self.hyperParams['-k'].upper()]
			if '-s' in self.hyperParams and type(self.hyperParams['-s']) is str:
				self.hyperParams['-s'] = SVM_TYPE_LOOKUP[self.hyperParams['-s'].upper()]
		
		
		
		#Create model
		
		#ThunderSVM
		if modelType == 'THUNDERSVM':
			self.model = ThunderSVM(self.hyperParams)
			self.modelType = 'ThunderSVM'
				
		#libsvm
		elif modelType == 'LIBSVM':
			params = ['-b',1]
			for item in ['-s','-t','-d','-g','-r','-c','-n','-p','-m','-e','-h','-wi','-q']:
				if item in self.hyperParams:
					params.append(item)
					params.append(self.hyperParams[item])
			self.model = LibSVM(params)
			self.modelType = 'LibSVM'
			
		
		elif modelType in ['SGD','SGDCLASSIFIER']:
			self.modelType='SGD'
			loss = 'log' if 'loss' not in self.hyperParams or self.hyperParams['loss'] == 'modified_huber' else 'modified_huber'
			self.model = SGDClassifier(loss=loss,
										penalty = self.hyperParams.get('penalty','l2'),
										alpha = self.hyperParams.get('alpha',0.001),
										l1_ratio = self.hyperParams.get('l1_ratio',0.15),
										fit_intercept = self.hyperParams.get('fit_intercept',True),
										max_iter = self.hyperParams.get('-iter',True),
										tol = self.hyperParams.get('-tol',True),
										shuffle = self.hyperParams.get('shuffle',True),
										verbose = self.hyperParams.get('-verbose',True),
										epsilon = self.hyperParams.get('-eps',0.1),
										n_jobs = self.hyperParams.get('-jobs',None),
										random_state = self.hyperParams.get('seed',1),
										learning_rate = self.hyperParams.get('lr','optimal'),
										eta0 = self.hyperParams.get('eta0',0.0),
										power_t=self.hyperParams.get('power_t',0.5),
										early_stopping=self.hyperParams.get('early_stopping',False),
										validation_fraction=self.hyperParams.get('validation_fraction',0.1),
										n_iter_no_change=self.hyperParams.get('n_iter_no_change',5),
										class_weight=self.hyperParams.get('-w',None),
										warm_start=self.hyperParams.get('warm_start',False),
										average=self.hyperParams.get('average',False))
				
		
		
		elif modelType in ['LINEARSVC','LIBLINEAR']:
			self.modelType = 'LinearSVC'
			self.model = CalibratedClassifierCV(
					method=self.hyperParams.get('CC_Method','Sigmoid'),
					n_jobs = self.hyperParams.get('CC_Jobs',None),
					ensemble = self.hyperParams.get('CC_Ensemble',True),
					base_estimator = LinearSVC(
						penalty = self.hyperParams.get('penalty','l2'),
						loss = self.hyperParams.get('loss','squared_hinge'),
						dual = self.hyperParams.get('dual',True),
						tol = self.hyperParams.get('-tol',True),
						C = self.hyperParams.get('-c',1.0),
						fit_intercept = self.hyperParams.get('fit_intercept',True),
						intercept_scaling = self.hyperParams.get('intercept_scaling',1),
						class_weight=self.hyperParams.get('-w',None),
						verbose = self.hyperParams.get('-verbose',True),
						random_state = self.hyperParams.get('seed',1),
						max_iter = self.hyperParams.get('-iter',True),
						))
			
		
		
		elif modelType in ['SVC']:
			self.modelType = 'SVC'
			self.model = SVC(
						C = self.hyperParams.get('-c',1.0),
						kernel=self.hyperParams.get('-k','rbf'),
						degree=self.hyperParams.get('-d',3),
						gamma=self.hyperParams.get('-g','scale'),
						coef0=self.hyperParams.get('-c0',0.0),
						shrinking=self.hyperParams.get('-shrink',True),
						probability=True,
						tol=self.hyperParams.get('-tol',0.001),
						cache_size=self.hyperParams.get('-cs',200),
						class_weight=self.hyperParams.get('-w',None),
						verbose = self.hyperParams.get('-verbose',True),
						max_iter = self.hyperParams.get('-iter',True),
						decision_function_shape=self.hyperParams.get('-df','ovr'),
						break_ties=self.hyperParams.get('break_ties',False),
						random_state = self.hyperParams.get('seed',1))
			
		else:
			print('Error, invalid model type')
			exit(42)



	def fit(self,trainFeatures,trainClasses):
		return self.model.fit(trainFeatures,trainClasses)
				
	def predict_proba(self,trainFeatures):
		return self.model.predict_proba(trainFeatures)
		
	def saveModel(self,fname):
		self.saveModelToFile(fname)
	def loadModel(self,fname):
		self.loadModelFromFile(fname)
	def saveModelToFile(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		if self.modelType in ['SVC','LinearSVC','SGD']:
			dump(self.model,fname)
		else:
			self.model.save(fname)

	def loadModelFromFile(self,fname):
		if self.modelType in ['SVC','LinearSVC','SGD']:
			self.model = load(fname)
		else:
			self.model.load(fname)