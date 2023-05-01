import numpy as np 
from etaCycle import EtaCycle

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def ReLU(x):
	""" Standard definition of relu """
	return np.maximum(x, 0) #to be checked
	
class Model:
	def __init__(self,hidden_layers, weight_decay_parameter, batch_size, lr_cycle_magnitude, size_of_dataset):
		# hidden_layers : array containing size of each hidden layer. ex [50] or [15,15]
		# lambda : weight decay parameter
		# batch size : number of point to take at each batch
		# lr_cycle_magnitude : parameter for the cyclical learning rate
		# size of dataset : size of the training dataset

		# inhererent parameters known from the dataset
		d = 1024*3 
		K = 10 
		layers_sizes = [d] + hidden_layers + [K] 

		ns = lr_cycle_magnitude*(size_of_dataset/batch_size) # 900 for 45000, 200 for 10000
		# 1 "cycle" means one up --> and one down <--. 
		# if lr_cycle_magnitude of ns = 2, 1 cycle means 4 epochs so 2 cycle means 8 epochs
		# if lr_cycle_magnitude of ns = 5, 1 cycle means 10 epochs 
		# if lr_cycle_magnitude of ns = 8, 1 cycle means 16 epochs so 2 cycle means 32 epochs
		self.layers = []
		for l in range(len(layers_sizes - 1)):
			self.layers.append({ "weight":np.random.normal(0,1/np.sqrt(layers_sizes[l]), (layers_sizes[l+1], layers_sizes[l])), "bias": np.random.normal(0, 0, (layers_sizes[l+1], 1))})
		self.weight_decay_parameter = weight_decay_parameter
		self.etaCycle = EtaCycle(1e-5, 1e-1, ns)
	
		
	def evaluate(self, X):
		X = X.T
		for l in range(len(self.layers)-1):
			X = ReLU(np.add(np.matmul(self.layers[0]["weight"], X), self.layers[0]["bias"])) 
		P = softmax(np.add(np.matmul(self.layers[1]["weight"], X), self.layers[1]["bias"]))
		
		return H, P

	def computeCost(self,Y,P,andAccuracy=True):
		'''
		input: dataloader, weigth, bias, weight decay parameter, and wheter to compute or not the accuracy
		returns the loss, cost(+weight decay) and accuracy
		'''
		batch_size = Y.shape[1]
		
		loss, cost, accuracy = 0,0, 0
		for im_idx in range(batch_size):
			loss -= np.matmul(Y[:,im_idx], np.log(P[:,im_idx]))
		loss /= batch_size

		weight_decay_sum = 0
		for layer in self.layers:
			weight_decay_sum += np.sum(np.square(layer["weight"]))
		cost = loss + self.weight_decay_parameter*weight_decay_sum 

		if(andAccuracy):
			accuracy = np.sum(np.argmax(P, 0) == np.argmax(Y, 0))/batch_size
		return loss, cost, accuracy

	def computeGradsNum(self, X, Y,P,h=1e-6):
		""" Converted from matlab code """
		grad_W1 = np.zeros(self.layers[0]["weight"].shape)
		grad_b1 = np.zeros(self.layers[0]["bias"].shape)
		grad_W2 = np.zeros(self.layers[1]["weight"].shape)
		grad_b2 = np.zeros(self.layers[1]["bias"].shape)

		X = X.T

		c, _, _ = self.computeCost(Y, P, andAccuracy=False)
		
		for i in range(grad_W1.shape[0]):
			for j in range(grad_W1.shape[1]):
				W_try = np.array(self.layers[0]["weight"])
				W_try[i,j] += h

				H = ReLU(np.add(np.matmul(W_try, X), self.layers[0]["bias"])) 
				P_try = softmax(np.add(np.matmul(self.layers[1]["weight"], H), self.layers[1]["bias"])) 
				c2, _, _ = self.computeCost(Y, P_try, andAccuracy=False)
				grad_W1[i,j] = (c2-c) / h
		
		for i in range(grad_W2.shape[0]):
			for j in range(grad_W2.shape[1]):
				W_try = np.array(self.layers[1]["weight"])
				W_try[i,j] += h

				H = ReLU(np.add(np.matmul(self.layers[0]["weight"], X), self.layers[0]["bias"])) 
				P = softmax(np.add(np.matmul(W_try, H), self.layers[1]["bias"])) 
				c2, _, _ = self.computeCost(Y, P, andAccuracy=False)
				grad_W2[i,j] = (c2-c) / h
		
		for i in range(grad_b1.shape[0]):
			b_try = np.array(self.layers[0]["bias"])
			b_try[i] += h
			H = ReLU(np.add(np.matmul(self.layers[0]["weight"], X), b_try)) 
			P = softmax(np.add(np.matmul(self.layers[1]["weight"], H), self.layers[1]["bias"])) 
			c2, _, _ = self.computeCost(Y, P, andAccuracy=False)
			grad_b1[i] = (c2-c) / h
		
		for i in range(grad_b2.shape[0]):
			b_try = np.array(self.layers[1]["bias"])
			b_try[i] += h
			H = ReLU(np.add(np.matmul(self.layers[0]["weight"], X), self.layers[0]["bias"])) 
			P = softmax(np.add(np.matmul(self.layers[1]["weight"], H), b_try)) 
			
			c2, _, _ = self.computeCost(Y, P, andAccuracy=False)
			grad_b2[i] = (c2-c) / h
		

		return [grad_W1, grad_b1, grad_W2, grad_b2]

	def computeGradsAnalytical(self,X,Y,H,P):
		grad_W1 = np.zeros(self.layers[0]["weight"].shape)
		grad_b1 = np.zeros(self.layers[0]["bias"].shape)
		grad_W2 = np.zeros(self.layers[1]["weight"].shape)
		grad_b2 = np.zeros(self.layers[1]["bias"].shape)

		G = -(Y-P)

		grad_W2 = np.matmul(G,H.T)/X.shape[0] + 2*self.weight_decay_parameter*self.layers[1]["weight"]
		grad_b2 = np.average(G,1).reshape(grad_b2.shape)

		G = np.matmul(self.layers[1]["weight"].T, G)
		G = np.multiply(G, np.where(H>0,1,0))

		grad_W1 = np.matmul(G, X)/X.shape[0] + 2*self.weight_decay_parameter*self.layers[0]["weight"]
		grad_b1 = np.average(G, 1).reshape(grad_b1.shape)
		return [grad_W1, grad_b1, grad_W2, grad_b2]
	
	def backpropagate(self, X, Y, H, P):
		self.etaCycle.next()
		
		[grad_W1, grad_b1, grad_W2, grad_b2] = self.computeGradsAnalytical(X,Y,H,P)
		self.layers[0]["weight"] -= self.etaCycle.eta*grad_W1
		self.layers[0]["bias"] -= self.etaCycle.eta*grad_b1
		self.layers[1]["weight"] -= self.etaCycle.eta*grad_W2
		self.layers[1]["bias"] -= self.etaCycle.eta*grad_b2
