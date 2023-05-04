import numpy as np 
from etaCycle import EtaCycle

epslion = 1e-6
alpha = 0.1
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
			self.layers.append(
				{ 
					"weight":np.random.normal(0,1/np.sqrt(layers_sizes[l]), (layers_sizes[l+1], layers_sizes[l])), 
					"bias": np.random.normal(0, 0, (layers_sizes[l+1], 1)),
					"scale":np.random.normal(0,1/np.sqrt(layers_sizes[l+1]), (layers_sizes[l+1],1)),
					"shift":np.random.normal(0,1/np.sqrt(layers_sizes[l+1]), (layers_sizes[l+1],1)),
					"mean": np.zeros((layers_sizes[l+1],1)),
					"variance": np.zeros((layers_sizes[l+1],1))
			})

		#remove useless info for last layer
		self.layers[-1].pop("scale")
		self.layers[-1].pop("shift")
		self.layers[-1].pop("mean")
		self.layers[-1].pop("variance")

		self.weight_decay_parameter = weight_decay_parameter
		self.etaCycle = EtaCycle(1e-5, 1e-1, ns)

		print(f'model \n size of layers : {layers_sizes} \n weight decay parameter : {weight_decay_parameter} \n')
	
	def evaluate(self, X, step):
		'''
		X in the input, the flatten image
		step is training or testing time 
		During testing, the un-normalized scores are normalized by known pre-computed means and variances that have been estimated during training
		During training, we computed them directly from the batch
		'''
		X = X.T
		metrics = [(0,0,X,0,0)]
		for layer in self.layers[0:-1]:
			S = np.add(np.matmul(layer["weight"], X), layer["bias"])
			
			if step=="training_time":
				# compute batch normalization
				mean, var = np.average(S, 1), np.var(S, 1)

				if np.array_equal(layer["mean"], np.zeros(layer["mean"].shape)):
					layer["mean"] = mean
					layer["variance"] = var

				# same mean and variance for test time
				layer["mean"] = alpha*layer["mean"] + (1-alpha)*mean
				layer["variance"] = alpha*layer["variance"] + (1-alpha)*var

			elif step=="testing_time":
				mean = layer["mean"]
				var = layer["variance"]
			S_hat = np.divide(S-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)) + epslion)
			S_wave = np.add(np.multiply(S_hat, layer["scale"]), layer["shift"])

			X = ReLU(S_wave)

			metrics.append((S, S_hat, X, mean, var))

		P = softmax(np.add(np.matmul(self.layers[-1]["weight"], X), self.layers[-1]["bias"]))

		# metrics has to be unpack into separate lists

		return metrics, P


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

	def computeGradsNum(self, X, Y,P,h, analytical):
		""" Converted from matlab code """
		grad_W1 = np.zeros(self.layers[0]["weight"].shape)
		grad_b1 = np.zeros(self.layers[0]["bias"].shape)

		X = X.T
		Pcopy = np.copy(P)

		c, _, _ = self.computeCost(Y, P, andAccuracy=False)
		
		for i in range(grad_W1.shape[0]):
			for j in range(grad_W1.shape[1]):
				W_try = np.array(self.layers[0]["weight"])
				W_try[i,j] += h
				Xtry = np.copy(X)

				Stry = np.add(np.matmul(W_try, Xtry), self.layers[0]["bias"])
				mean, var = np.average(Stry, 1), np.var(Stry, 1)
				Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)) + epslion)
				Stry_wave = np.add(np.multiply(Stry_hat, self.layers[0]["scale"]), self.layers[0]["shift"])
				Xtry = ReLU(Stry_wave)

				for layer in self.layers[1:-1]:
					Stry = np.add(np.matmul(layer["weight"], Xtry), layer["bias"])
					mean, var = np.average(Stry, 1), np.var(Stry, 1)
					Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)) + epslion)
					Stry_wave = np.add(np.multiply(Stry_hat, layer["scale"]), layer["shift"])
					Xtry = ReLU(Stry_wave)
				Ptry = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
				c2, _, _ = self.computeCost(Y, Ptry, andAccuracy=False)
				grad_W1[i,j] = (c2-c) / h
		
		
		for i in range(grad_b1.shape[0]):
			b_try = np.array(self.layers[0]["bias"])
			b_try[i] += h
			Xtry = np.copy(X)

			Stry = np.add(np.matmul(self.layers[0]["weight"], Xtry), b_try)
			mean, var = np.average(Stry, 1), np.var(Stry, 1)
			Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)) + epslion)
			Stry_wave = np.add(np.multiply(Stry_hat, self.layers[0]["scale"]), self.layers[0]["shift"])
			Xtry = ReLU(Stry_wave)

			for layer in self.layers[1:-1]:
				Stry = np.add(np.matmul(layer["weight"], Xtry), layer["bias"])
				mean, var = np.average(Stry, 1), np.var(Stry, 1)
				Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)) + epslion)
				Stry_wave = np.add(np.multiply(Stry_hat, layer["scale"]), layer["shift"])
				Xtry = ReLU(Stry_wave)
			Ptry = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
			c2, _, _ = self.computeCost(Y, Ptry, andAccuracy=False)
			grad_b1[i] = (c2-c) / h
		
		

		return [grad_W1, grad_b1]

	def computeGradsAnalytical(self,X,Y,P, metrics):

		#unzip metrics
		Ss, S_hats, Xs, Ms, Vs = zip(*metrics)
		
		N = X.shape[0] #batch size
		G = -(Y-P)
		
		last_layer = self.layers[-1]
		grad_weight = np.matmul(G,Xs[-1].T)/N + 2*self.weight_decay_parameter*last_layer["weight"]
		grad_bias = np.average(G,1).reshape(last_layer["bias"].shape)
		grads = [[grad_weight, grad_bias]] # to be appended later

		G = np.matmul(last_layer["weight"].T, G)
		G = np.multiply(G, np.where(Xs[-1]>0,1,0))
		

		for layer_id, layer in enumerate(reversed(self.layers[0:-1])):
			if layer_id != len(self.layers)-1:
				grad_scale = np.average(np.multiply(G, S_hats[-1-layer_id]), 1).reshape(layer["scale"].shape)
				grad_shift = np.average(G,1).reshape(layer["shift"].shape)

				# Propagate the gradients through the scale and shift
				G = np.multiply(G, layer["scale"])

				# Propagate Gbatch through the batch normalization
				gamma1 = np.power(Vs[-1-layer_id]+epslion, -0.5)
				gamma2 = np.power(Vs[-1-layer_id]+epslion, -1.5)

				G1 = np.multiply(G, gamma1.reshape(gamma1.shape[0],1))
				G2 = np.multiply(G, gamma2.reshape(gamma2.shape[0],1))
				D = np.subtract(Ss[-1-layer_id], Ms[-1-layer_id].reshape(Ms[-1-layer_id].shape[0], 1))

				c = np.sum(np.multiply(G2,D), 1)
				c = c.reshape(c.shape[0],1)
				G = np.subtract(np.subtract(G1, np.average(G1,1).reshape(G1.shape[0],1)), np.multiply(D,np.matmul(c,np.ones((1,N))))/N)
				#G1 - np.matmul(np.sum(G1,1).reshape(12,1),np.ones((1,10)))/N - np.multiply(D,np.matmul(c.reshape(12,1), np.ones((1,10))))/N

				grad_weight = np.matmul(G, Xs[-1-layer_id-1].T)/N + 2*self.weight_decay_parameter*layer["weight"]
				grad_bias = np.average(G,1).reshape(layer["bias"].shape)
				G = np.matmul(layer["weight"].T, G)
				G = np.multiply(G, np.where(Xs[-1-layer_id-1]>0,1,0))

			grads.append([grad_weight, grad_bias, grad_scale, grad_shift])
			
		grads.reverse()
		return grads


	def backpropagate(self, X, Y, P, metrics):

		self.etaCycle.next()
		grads = self.computeGradsAnalytical(X,Y,P, metrics)
		for layer_id, layer in enumerate(self.layers):
			layer["weight"] -= self.etaCycle.eta*grads[layer_id][0]
			layer["bias"] -= self.etaCycle.eta*grads[layer_id][1]
			if(len(grads[layer_id])>2):
				layer["scale"] -= self.etaCycle.eta*grads[layer_id][2]
				layer["shift"] -= self.etaCycle.eta*grads[layer_id][3]
			