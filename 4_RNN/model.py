import numpy as np 
from dataset import Data
def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)


class Model:
	def __init__(self, data, m=100):
		'''
		data: dataloader
		m: dimensionality of hidden layers
		'''
		self.data = data
		self.K = data.K
		self.m = m

		self.eta = .1
		self.seq_length = 25
		# inhererent parameters known from the dataset
		sig = 0.01
		self.RNN = { 
					"U":np.random.normal(0,sig,(m,self.K)), 
					"W":np.random.normal(0,sig,(m,m)), 
					"V":np.random.normal(0,sig,(self.K,m)), 
					"b":np.zeros((m,1)),
					"c":np.zeros((self.K,1))}
		

	def evaluate_test(self, h0, x0, n):
		res = np.zeros((x0.shape[0],n))
		res[:,0] = x0.T
		h = h0
		x = x0
		for i in range(1,n):
			a = np.matmul(self.RNN["W"], h) + np.matmul(self.RNN["U"], x) + self.RNN["b"]
			h = np.tanh(a)
			o = np.matmul(self.RNN["V"], h) + self.RNN["c"]
			p = softmax(o)

			#build new x
			x = np.zeros(p.shape)
			x[np.argmax(p)] = 1
			res[:,i] = x.T
		print(self.data.vec_to_seq(res))
		return res
	

	def computeLoss(self,output, pred):
		loss = 0
		n = output.shape[1]

		for char_id in range(n):
			loss -= np.matmul(output[:,char_id], np.log(pred[:,char_id]))
		return loss
	

	def forward(self, h0, input, output,  n):
		pred = np.zeros(input.shape)
		a_s = np.zeros((h0.shape[0], n))
		h_s = np.zeros((h0.shape[0], n+1))
		h = h0
		h_s[:,0] = h0.T
		
		for i in range(0,n):
			x = input[:,i]
			a = np.matmul(self.RNN["W"], h) + np.matmul(self.RNN["U"], x.reshape(x.shape[0], 1)) + self.RNN["b"]
			h = np.tanh(a)
			o = np.matmul(self.RNN["V"], h) + self.RNN["c"]
			p = softmax(o)

			#in forward, does not necessary want to build y, can keep to the probability right ? 
			y = np.zeros(p.shape)
			y[np.argmax(p)] = 1
			pred[:,i] = p.T
			a_s[:,i] = a.T
			h_s[:,i+1] = h.T
		loss = self.computeLoss(output, pred)
		return a_s, h_s, pred, loss
		
	def computeGrads(self, input, output, a_s, h_s, pred):
		grad_V = np.zeros(self.RNN["V"].shape)
		grad_W = np.zeros(self.RNN["W"].shape)
		grad_U = np.zeros(self.RNN["U"].shape)
		grad_b = np.zeros(self.RNN["b"].shape)
		grad_c = np.zeros(self.RNN["c"].shape)


		#for t = tmax
		grad_ot = np.expand_dims(np.subtract(pred[:,-1], output[:,-1]), 1)
		
		grad_V = np.matmul(grad_ot, np.expand_dims(h_s[:,-1],1).T)
		grad_c = grad_ot

		grad_ht = np.matmul(grad_ot.T,self.RNN["V"])
		grad_at = np.matmul(grad_ht, np.diag(1-np.power(a_s[:,-1], 2))) 
		
		grad_W = np.matmul(grad_at.T, np.expand_dims(h_s[:,-2],1).T)
		grad_U = np.matmul(grad_at.T, np.expand_dims(input[:,-1],1).T)
		grad_b = grad_at
		for t in range(-1,25,-1):
			A = 2



		return [grad_V, grad_W, grad_U, grad_b, grad_c]

	# def computeGradsNum(self, X, Y,P,h):
	# 	""" Converted from matlab code """
	# 	grad_W1 = np.zeros(self.layers[0]["weight"].shape)
	# 	grad_b1 = np.zeros(self.layers[0]["bias"].shape)

	# 	X = X.T

	# 	c, _, _ = self.computeCost(Y, P, andAccuracy=False)
	# 	if(self.batch_normalization):
	# 		for i in range(grad_W1.shape[0]):
	# 			for j in range(grad_W1.shape[1]):
	# 				W_try = np.array(self.layers[0]["weight"])
	# 				W_try[i,j] += h
	# 				Xtry = np.copy(X)

	# 				Stry = np.add(np.matmul(W_try, Xtry), self.layers[0]["bias"])
	# 				mean, var = np.average(Stry, 1), np.var(Stry, 1)
	# 				Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1) + epslion ))
	# 				Stry_wave = np.add(np.multiply(Stry_hat, self.layers[0]["scale"]), self.layers[0]["shift"])
	# 				Xtry = ReLU(Stry_wave)

	# 				for layer in self.layers[1:-1]:
	# 					Stry = np.add(np.matmul(layer["weight"], Xtry), layer["bias"])
	# 					mean, var = np.average(Stry, 1), np.var(Stry, 1)
	# 					Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)+ epslion))
	# 					Stry_wave = np.add(np.multiply(Stry_hat, layer["scale"]), layer["shift"])
	# 					Xtry = ReLU(Stry_wave)
	# 				Ptry = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
	# 				c2, _, _ = self.computeCost(Y, Ptry, andAccuracy=False)
	# 				grad_W1[i,j] = (c2-c) / h
			
		
	# 		for i in range(grad_b1.shape[0]):
	# 			b_try = np.array(self.layers[0]["bias"])
	# 			b_try[i] += h
	# 			Xtry = np.copy(X)

	# 			Stry = np.add(np.matmul(self.layers[0]["weight"], Xtry), b_try)
	# 			mean, var = np.average(Stry, 1), np.var(Stry, 1)
	# 			Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)+ epslion))
	# 			Stry_wave = np.add(np.multiply(Stry_hat, self.layers[0]["scale"]), self.layers[0]["shift"])
	# 			Xtry = ReLU(Stry_wave)

	# 			for layer in self.layers[1:-1]:
	# 				Stry = np.add(np.matmul(layer["weight"], Xtry), layer["bias"])
	# 				mean, var = np.average(Stry, 1), np.var(Stry, 1)
	# 				Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)+epslion))
	# 				Stry_wave = np.add(np.multiply(Stry_hat, layer["scale"]), layer["shift"])
	# 				Xtry = ReLU(Stry_wave)
	# 			Ptry = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
	# 			c2, _, _ = self.computeCost(Y, Ptry, andAccuracy=False)
	# 			grad_b1[i] = (c2-c) / h
			
	# 	else:
	# 		for i in range(grad_W1.shape[0]):
	# 			for j in range(grad_W1.shape[1]):
	# 				W_try = np.array(self.layers[0]["weight"])
	# 				W_try[i,j] += h
	# 				Xtry = np.copy(X)
	# 				Xtry = ReLU(np.add(np.matmul(W_try, Xtry), self.layers[0]["bias"])) 
	# 				for layer in self.layers[1:-1]:
	# 					Xtry = ReLU(np.add(np.matmul(layer["weight"], Xtry), layer["bias"])) 
	# 				P = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
	# 				c2, _, _ = self.computeCost(Y, P, andAccuracy=False)
	# 				grad_W1[i,j] = (c2-c) / h
		
		
	# 		for i in range(grad_b1.shape[0]):
	# 			b_try = np.array(self.layers[0]["bias"])
	# 			b_try[i] += h
	# 			Xtry = np.copy(X)
	# 			Xtry = ReLU(np.add(np.matmul(self.layers[0]["weight"],Xtry), b_try)) 
	# 			for layer in self.layers[1:-1]:
	# 				Xtry = ReLU(np.add(np.matmul(layer["weight"], Xtry), layer["bias"])) 
	# 			P = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
	# 			c2, _, _ = self.computeCost(Y, P, andAccuracy=False)
	# 			grad_b1[i] = (c2-c) / h

	# 	return [grad_W1, grad_b1]

	# def computeGradsAnalytical(self,X,Y,P, metrics):
	# 	N = X.shape[0] #batch size
	# 	#unzip metrics
	# 	Ss, S_hats, Xs = zip(*metrics)
	# 	if(self.batch_normalization):
	# 		G = -(Y-P)
	# 		last_layer = self.layers[-1]
	# 		grad_weight = np.matmul(G,Xs[-1].T)/N + 2*self.weight_decay_parameter*last_layer["weight"]
	# 		grad_bias = np.average(G,1).reshape(last_layer["bias"].shape)
	# 		grads = [[grad_weight, grad_bias]] # to be appended later

	# 		G = np.matmul(last_layer["weight"].T, G)
	# 		G = np.multiply(G, np.where(Xs[-1]>0,1,0))
			

	# 		for layer_id, layer in enumerate(reversed(self.layers[0:-1])):
	# 			if layer_id != len(self.layers)-1:
	# 				grad_scale = np.average(np.multiply(G, S_hats[-1-layer_id]), 1).reshape(layer["scale"].shape)
	# 				grad_shift = np.average(G,1).reshape(layer["shift"].shape)

	# 				# Propagate the gradients through the scale and shift
	# 				G = np.multiply(G, layer["scale"])				
					
	# 				# new implementation
	# 				mu, var = layer["batch_normalization"].getMuAndVar("training_time")
	# 				gamma1 = np.power(var+epslion, -0.5)
	# 				gamma2 = np.power(var+epslion, -1.5)

	# 				G1 = np.multiply(G, gamma1)
	# 				G2 = np.multiply(G, gamma2)
	# 				D = np.subtract(Ss[-1-layer_id], mu)
	# 				c = np.sum(np.multiply(G2,D), 1)
	# 				c = c.reshape(c.shape[0],1)
	# 				G = np.subtract(np.subtract(G1, np.average(G1,1).reshape(G1.shape[0],1)), np.multiply(D,np.matmul(c,np.ones((1,N))))/N)

	# 				grad_weight = np.matmul(G, Xs[-1-layer_id-1].T)/N + 2*self.weight_decay_parameter*layer["weight"]
	# 				grad_bias = np.average(G,1).reshape(layer["bias"].shape)
	# 				G = np.matmul(layer["weight"].T, G)
	# 				G = np.multiply(G, np.where(Xs[-1-layer_id-1]>0,1,0))

	# 			grads.append([grad_weight, grad_bias, grad_scale, grad_shift])
		
	# 	else:
	# 		grads = []
	# 		G = -(Y-P)
			
	# 		for layer_id, layer in enumerate(reversed(self.layers)):
	# 			if layer_id != len(self.layers)-1:
	# 				grad_weight = np.matmul(G,Xs[-1-layer_id].T)/N + 2*self.weight_decay_parameter*layer["weight"]
	# 				grad_bias = np.average(G,1).reshape(layer["bias"].shape)
	# 				G = np.matmul(layer["weight"].T, G)
	# 				G = np.multiply(G, np.where(Xs[-1-layer_id]>0,1,0))
	# 			else:
	# 				#computing gradients for the first layer
	# 				grad_weight = np.matmul(G,X)/N + 2*self.weight_decay_parameter*layer["weight"]
	# 				grad_bias = np.average(G,1).reshape(layer["bias"].shape)
	# 			grads.append([grad_weight, grad_bias])
				
	# 	grads.reverse()
	# 	return grads


	def backpropagate(self, input, output, pred, h_s):
		grads = self.computeGrads(input, output, pred, h_s)
		self.RNN["V"] += self.eta*grads[0]

		# not that thing exactly.
data = Data()
model = Model(data)

# debug purposes
A = ["a", "l", "a", "n"]
vec = data.seq_to_vec(A)

# evaluate test
x0 = "a"
x0 = data.seq_to_vec(x0)
h0 = np.ones((100,1))
model.evaluate_test(h0, x0, 25)

#forward
input = data.seq_to_vec(data.book_data[0:25])
output = data.seq_to_vec(data.book_data[1:26])
h0 = np.zeros((100,1))
a_s, h_s, pred, loss = model.forward(h0, input, output, 25)
model.computeGrads(input, output, a_s, h_s, pred)
A = 2
