import numpy as np 
from etaCycle import EtaCycle

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def ReLU(x):
	""" Standard definition of relu """
	return np.maximum(x, 0) #to be checked
	
class Model:
	def __init__(self,hidden_layers_structure, weight_decay_parameter, batch_size, lr_cycle_magnitude, size_of_dataset):
		# hidden_layers_structure : array containing size of each hidden layer. ex [50] or [15,15]
		# lambda : weight decay parameter
		# batch size : number of point to take at each batch
		# lr_cycle_magnitude : parameter for the cyclical learning rate
		# size of dataset : size of the training dataset

		# inhererent parameters known from the dataset
		d = 1024*3 
		K = 10 
		layers_sizes = [d] + hidden_layers_structure + [K] 

		ns = lr_cycle_magnitude*(size_of_dataset/batch_size) # 900 for 45000, 200 for 10000
		# 1 "cycle" means one up --> and one down <--. 
		# if lr_cycle_magnitude of ns = 2, 1 cycle means 4 epochs so 2 cycle means 8 epochs
		# if lr_cycle_magnitude of ns = 5, 1 cycle means 10 epochs 
		# if lr_cycle_magnitude of ns = 8, 1 cycle means 16 epochs so 2 cycle means 32 epochs
		self.layers = []
		for l in range(len(layers_sizes) - 1):
			self.layers.append({ "weight":np.random.normal(0,1/np.sqrt(layers_sizes[l]), (layers_sizes[l+1], layers_sizes[l])), "bias": np.random.normal(0, 0, (layers_sizes[l+1], 1))})
		self.weight_decay_parameter = weight_decay_parameter
		self.etaCycle = EtaCycle(1e-5, 1e-1, ns)

		print(f'model \n size of layers : {layers_sizes} \n weight decay parameter : {weight_decay_parameter} \n')
	
		
	def evaluate(self, X):
		X = X.T
		Hs = []
		for layer in self.layers[0:-1]:
			X = ReLU(np.add(np.matmul(layer["weight"], X), layer["bias"])) 
			Hs.append(X)
		P = softmax(np.add(np.matmul(self.layers[-1]["weight"], X), self.layers[-1]["bias"]))
		
		return Hs, P

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

		X = X.T

		c, _, _ = self.computeCost(Y, P, andAccuracy=False)
		
		for i in range(grad_W1.shape[0]):
			for j in range(grad_W1.shape[1]):
				W_try = np.array(self.layers[0]["weight"])
				W_try[i,j] += h
				Xtry = np.copy(X)
				Xtry = ReLU(np.add(np.matmul(W_try, Xtry), self.layers[0]["bias"])) 
				for layer in self.layers[1:-1]:
					Xtry = ReLU(np.add(np.matmul(layer["weight"], Xtry), layer["bias"])) 
				P = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
				c2, _, _ = self.computeCost(Y, P, andAccuracy=False)
				grad_W1[i,j] = (c2-c) / h
		
		
		for i in range(grad_b1.shape[0]):
			b_try = np.array(self.layers[0]["bias"])
			b_try[i] += h
			Xtry = np.copy(X)
			Xtry = ReLU(np.add(np.matmul(self.layers[0]["weight"],Xtry), b_try)) 
			for layer in self.layers[1:-1]:
				Xtry = ReLU(np.add(np.matmul(layer["weight"], Xtry), layer["bias"])) 
			P = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
			c2, _, _ = self.computeCost(Y, P, andAccuracy=False)
			grad_b1[i] = (c2-c) / h
		
		

		return [grad_W1, grad_b1]

	def computeGradsAnalytical(self,X,Y,Hs,P):
		grad_W1 = np.zeros(self.layers[0]["weight"].shape)
		grad_b1 = np.zeros(self.layers[0]["bias"].shape)
		grad_W2 = np.zeros(self.layers[1]["weight"].shape)
		grad_b2 = np.zeros(self.layers[1]["bias"].shape)

		grads = []
		N = X.shape[0]
		G = -(Y-P)
		
		for layer_id, layer in enumerate(reversed(self.layers)):
			if layer_id != len(self.layers)-1:
				grad_weight = np.matmul(G,Hs[-1-layer_id].T)/N + 2*self.weight_decay_parameter*layer["weight"]
				grad_bias = np.average(G,1).reshape(layer["bias"].shape)
				G = np.matmul(layer["weight"].T, G)
				G = np.multiply(G, np.where(Hs[-1-layer_id]>0,1,0))
			else:
				#computing gradients for the first layer
				grad_weight = np.matmul(G,X)/N + 2*self.weight_decay_parameter*layer["weight"]
				grad_bias = np.average(G,1).reshape(layer["bias"].shape)
			grads.append([grad_weight, grad_bias])
			


		grads.reverse()
		return grads
	
	def backpropagate(self, X, Y, Hs, P):
		self.etaCycle.next()
		grads = self.computeGradsAnalytical(X,Y,Hs,P)
		for layer_id, layer in enumerate(self.layers):
			layer["weight"] -= self.etaCycle.eta*grads[layer_id][0]
			layer["bias"] -= self.etaCycle.eta*grads[layer_id][1]
			