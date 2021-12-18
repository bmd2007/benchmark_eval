
from SimpleTorchDictionaryDataset import SimpleTorchDictionaryDataset
#designed for usage with neural network models using dictionary datasets
class GenericNetworkModel(object):
	def __init__(self,hyp={},fullGPU=False,deviceType=None,numEpochs=100,batchSize=256,optType='Adam',lr=1e-2,momentum=0,minLr=1e-6,schedFactor=.1,schedPatience=2,schedThresh=1e-2,sched_cooldown=0,weightDecay=0,seed=1):
	
		self.hyp = hyp
		self.fullGPU = hyp.get('fullGPU',fullGPU)
		self.numEpochs = hyp.get('numEpochs',numEpochs)
		self.seed = hyp.get('seed',seed)
		self.minLr = hyp.get('minLr',minLr)
		self.deviceType = hyp.get('deviceType',deviceType)

		#move network runner properties into hyperparams list if needed
		hyp['batchSize'] = hyp.get('batchSize',batchSize)
		hyp['optType'] = hyp.get('optType',optType)
		hyp['lr'] = hyp.get('lr',lr)
		hyp['seed'] = self.seed
		hyp['deviceType'] = self.deviceType
		hyp['schedFactor'] = hyp.get('schedFactor',schedFactor)
		hyp['schedPatience'] = hyp.get('schedPatience',schedPatience)
		hyp['schedThresh'] = hyp.get('schedThresh',schedThresh)
		hyp['schedCooldown'] = hyp.get('schedCooldown',sched_cooldown)
		hyp['weightDecay'] = hyp.get('weightDecay',weightDecay)
		hyp['momentum'] = hyp.get('momentum',momentum)
		
		self.model = None
	
	def saveModelToFile(self,fname):
		if self.model is None:
			print('Error, no model to save')
			exit(42)
		self.model.save(fname)
		
	def saveModel(self,fname):
		self.saveModelToFile(fname)
		
	def genModel(self):
		pass
		
	def loadModelFromFile(self,fname):
		if self.model is None:
			self.genModel()
		self.model.load(fname)
	
	def loadModel(self,fname):
		self.loadModelFromFile(fname)

	#train network
	def fit(self,pairLst,classes,dataMatrix,validationPairs=None, validationClasses=None):
		self.genModel()
		dataset = SimpleTorchDictionaryDataset(dataMatrix,pairLst,classes,full_gpu=self.fullGPU)
		if validationPairs is None:
			self.model.train(dataset, self.numEpochs,seed=self.seed,min_lr=self.minLr)
		else:
			validDataset = SimpleTorchDictionaryDataset(dataMatrix,validationPairs,validationClasses,full_gpu=self.fullGPU)
			self.model.trainWithValidation(dataset,validDataset,self.numEpochs,seed=self.seed,min_lr=self.minLr)
		
	#predict on network
	def predict_proba(self,pairLst,dataMatrix):
		dataset = SimpleTorchDictionaryDataset(dataMatrix,pairLst)
		probs,loss = self.model.predict(dataset)
		return probs
		
		
