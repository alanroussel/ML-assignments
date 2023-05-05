import numpy as np 

alpha = 0.9
class BatchNormalization:
	def __init__(self, shape):
		self.average_mu = np.zeros((shape))
		self.average_variance = np.zeros((shape))

		self.current_mu = np.zeros((shape))
		self.current_variance = np.zeros((shape))
		self.shape = shape
		self.init = True

	def compute(self,S):
		self.current_mu, self.current_variance = np.average(S, 1).reshape(self.shape), np.var(S, 1).reshape(self.shape)
		if self.init:
			self.average_mu = self.current_mu
			self.average_variance = self.current_variance
			self.init = False
		else:
			self.average_mu = alpha*self.average_mu + (1-alpha)*self.current_mu
			self.average_variance = alpha*self.average_variance + (1-alpha)*self.current_variance
	
	def getMuAndVar(self, step):
		if(step == "training_time"):
			return self.current_mu, self.current_variance
		elif(step=="testing_time"):
			return self.average_mu, self.average_variance