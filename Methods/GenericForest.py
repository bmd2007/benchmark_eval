from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier
from joblib import dump, load
from lightgbm import LGBMClassifier

#Supports SVC's for ThunderSVM, LibSVM, and Sklearn

class GenericForest(object):
	def __init__(self,hyperParams):
		
        #create a copy of the hyperparams
		self.hyperParams = hyperParams.copy()

		lst = [('learning_rate','lr'),
				('max_leaf_nodes','num_leaves'),
				('min_child_samples','min_samples_leaf'),
				('random_state','seed'),
				('verbose','silent')]
		#handle multiple aliases for forest data
		for tup in lst:
			for i in range(0,len(tup)):
				if tup[i] in self.hyperParams:
					for j in range(0,len(tup)):
						self.hyperParams[tup[j]] = self.hyperParams[tup[i]]
					break
				
		modelType = self.hyperParams.get('Model','RandomForest').upper()
		
		
		if modelType =='RANDOMFOREST':
			self.modelType = 'RANDFOREST'
			self.model = RandomForestClassifier(n_estimators=self.hyperParams.get('n_estimators',100),
										criterion = self.hyperParams.get('criterion','gini'),
										max_depth = self.hyperParams.get('max_depth',None),
										min_samples_split = self.hyperParams.get('min_samples_split',2),
										min_samples_leaf = self.hyperParams.get('min_samples_leaf',1),
										min_weight_fraction_leaf = self.hyperParams.get('min_weight_fraction_leaf',0),
										max_features = self.hyperParams.get('max_features','auto'),
										max_leaf_nodes = self.hyperParams.get('max_leaf_nodes',None),
										min_impurity_decrease = self.hyperParams.get('min_impurity_decrease',0),
										bootstrap = self.hyperParams.get('bootstrap',False),
										oob_score = self.hyperParams.get('oob_score',False),
										n_jobs = self.hyperParams.get('-jobs',-1),
										random_state = self.hyperParams.get('seed',1),
										verbose = self.hyperParams.get('-verbose',0),
										warm_start=self.hyperParams.get('warm_start',False),
										class_weight=self.hyperParams.get('class_weight',None),
										ccp_alpha=self.hyperParams.get('ccp_alpha',0),
										max_samples=self.hyperParams.get('max_samples',None))
		elif modelType == 'ROTATIONFOREST':
			self.modelType = 'ROTFOREST'
			self.model = RotationForestClassifier(n_estimators=self.hyperParams.get('n_estimators',10),
										criterion = self.hyperParams.get('criterion','gini'),
										n_features_per_subset = self.hyperParams.get('n_features_per_subset',3),
										rotation_algo=self.hyperParams.get('rotation_alog','pca'),
										max_depth = self.hyperParams.get('max_depth',None),
										min_samples_split = self.hyperParams.get('min_samples_split',2),
										min_samples_leaf = self.hyperParams.get('min_samples_leaf',1),
										min_weight_fraction_leaf = self.hyperParams.get('min_weight_fraction_leaf',0),
										max_features = self.hyperParams.get('max_features',1),
										max_leaf_nodes = self.hyperParams.get('max_leaf_nodes',None),
										bootstrap = self.hyperParams.get('bootstrap',False),
										oob_score = self.hyperParams.get('oob_score',False),
										n_jobs = self.hyperParams.get('-jobs',-1),
										random_state = self.hyperParams.get('seed',1),
										verbose = self.hyperParams.get('-verbose',0),
										warm_start=self.hyperParams.get('warm_start',False),
										class_weight=self.hyperParams.get('class_weight',None))
									
		elif modelType in ['LGBM','LIGHTGBM']:
			self.model = LGBMClassifier(boosting_type = self.hyperParams.get('boosting_type','gbdt'),
										num_leaves = self.hyperParams.get('num_leaves',11),
										max_depth = self.hyperParams.get('max_depth',None),
										learning_rate = self.hyperParams.get('lr',0.1),
										n_estimators=self.hyperParams.get('n_estimators',10),
										subsample_for_bin = self.hyperParams.get('subsample_for_bin',200000),
										objective = self.hyperParams.get('objective',None),
										class_weight=self.hyperParams.get('class_weight',None),
										min_split_gain = self.hyperParams.get('min_split_gain',0),
										min_child_weight = self.hyperParams.get('min_child_weight',0.001),
										min_child_samples=self.hyperParams.get('min_child_samples',20),
										subsample = self.hyperParams.get('subsample',1.0),
										subsample_freq = self.hyperParams.get('subsample_freq',0),
										reg_alpha=self.hyperParams.get('reg_alpha',0.0),
										reg_lambda=self.hyperParams.get('reg_lambda',0.0),
										random_state = self.hyperParams.get('seed',1),
										n_jobs = self.hyperParams.get('-jobs',-1),
										silent = self.hyperParams.get('verbose',True),
										importance_type=self.hyperParams.get('importance_type','split'))
		else:
			print('Error, invalid model type')
			exit(42)



	def fit(self,trainFeatures,trainClasses):
		z = self.model.fit(trainFeatures,trainClasses)
		return z
				
	def predict_proba(self,trainFeatures):
		return self.model.predict_proba(trainFeatures)
		
	def loadModel(self,fname):
		self.loadModelFromFile(fname)
	
	def saveModel(self,fname):
		self.saveModelToFile(fname)
		
	def saveModelToFile(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		dump(self.model,fname)

	def loadModelFromFile(self,fname):
		self.model = load(fname)
