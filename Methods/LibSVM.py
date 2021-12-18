import numpy as np
import libsvm.svmutil as libSVM
class LibSVM(object):
	def __init__(self,paramsLst):
		self.paramsLst = paramsLst
		self.svmModel = None
		#whether or not to flip the classes
		#this is a weird error, but, it seems like the svm doesn't remember the order of the classes
		#what this code currently does, in the fit section, is predict the training data,
		#compare where the positive interactions are in the those predictions compared to flipping the predictions
		#and determines if the order of the classes has been flipped or not, setting this variable to flip them
		#after each prediction if necessary.  If I can figure out why this is happening, I will remove the code and fix it.
		self.flipClasses = False

	def fit(self,features,classes):
		if len(classes.shape) > 1:
			classes = classes.flatten()
		self.svmModel = libSVM.svm_train(classes,features,self.paramsLst)
		
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
		results = libSVM.svm_predict([],features,self.svmModel,'-b 1')
		classifications = results[0]
		accuracy = results[1]
		probabilities = results[2]
		probabilities = np.asarray(probabilities)
		if self.flipClasses:
			probabilities = np.hstack((np.expand_dims(probabilities[:,1],axis=1),np.expand_dims(probabilities[:,0],axis=1)))

		return probabilities

	def save(self,fname):
		libSVM.svm_save_model(fname,self.svmModel)
		
	def load(self,fname):
		self.svmModel = libSVM.svm_load_model(fname)