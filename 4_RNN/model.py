import numpy as np 
from dataset import Data
from tqdm import tqdm

epslion = 1e-15


def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)


class Model:
	def __init__(self, data, eta, m=100):
		'''
		data: dataloader
		m: dimensionality of hidden layers
		'''
		self.data = data
		self.K = data.K
		self.m = m
		self.eta = eta
		self.seq_length = 25
		# inhererent parameters known from the dataset
		sig = 0.01
		self.RNN = { 
					"U":np.random.normal(0,sig,(m,self.K)), 
					"W":np.random.normal(0,sig,(m,m)), 
					"V":np.random.normal(0,sig,(self.K,m)), 
					"b":np.zeros((m,1)),
					"c":np.zeros((self.K,1))}

		self.AdaGrad_m = { 
					"U":np.zeros((m,self.K)), 
					"W":np.zeros((m,m)), 
					"V":np.zeros((self.K,m)), 
					"b":np.zeros((m,1)),
					"c":np.zeros((self.K,1))}
		
		print(f'init model w m = {m}')
		

	def synthesize(self,x0,h,n):
		res = np.zeros((x0.shape[0],n))
		res[:,0] = x0.T
		x = x0

		for i in range(1,n):
			a = np.matmul(self.RNN["W"], h) + np.matmul(self.RNN["U"], x) + self.RNN["b"]
			h = np.tanh(a)
			o = np.matmul(self.RNN["V"], h) + self.RNN["c"]
			p = softmax(o)

			#build new x

			# Compute the cumulative sum of p and draw a random sample from [0,1)
			cumulative_sum = np.cumsum(p)
			draw_number = np.random.sample()

			# Find the element that corresponds to this random sample
			pos = np.where(cumulative_sum > draw_number)[0][0]

			x = np.zeros(p.shape)
			x[pos] = 1
			res[:,i] = x.T

		return res
	

	def computeLoss(self,output, pred):
		loss = 0
		n = output.shape[1]

		for char_id in range(n):
			loss -= np.matmul(output[:,char_id], np.log(pred[:,char_id]))
		return loss
	

	def forward(self,input,h, weigth_to_consider=None):
		#init what we are going to store
		n = input.shape[1]
		pred = np.zeros(input.shape)

		if weigth_to_consider:
			U,W,V,b,c = weigth_to_consider.values()
		else:
			U,W,V,b,c = self.RNN.values()


		a_s = np.zeros((self.m, n))
		h_s = np.zeros((self.m, n+1))
		h_s[:,0] = h.T
		
		for i in range(0,n):
			x = input[:,i]
			a = np.matmul(W, h) + np.matmul(U, x.reshape(x.shape[0], 1)) + b
			h = np.tanh(a)
			o = np.matmul(V, h) + c
			p = softmax(o)

			#save
			pred[:,i] = p.T
			a_s[:,i] = a.T
			h_s[:,i+1] = h.T
		
		return a_s, h_s, pred

	
	def computeGrads(self, input, output, a_s, h_s, pred):
		grad_V = np.zeros(self.RNN["V"].shape)
		grad_W = np.zeros(self.RNN["W"].shape)
		grad_U = np.zeros(self.RNN["U"].shape)
		grad_b = np.zeros(self.RNN["b"].shape)
		grad_c = np.zeros(self.RNN["c"].shape)

		t_length = input.shape[1]

		#for t = tmax
		grad_ot = np.expand_dims(np.subtract(pred[:,-1], output[:,-1]), 1)
		
		grad_V = np.matmul(grad_ot, np.expand_dims(h_s[:,-1],1).T)
		grad_c = grad_ot

		grad_ht = np.matmul(grad_ot.T,self.RNN["V"])
		grad_at = np.matmul(grad_ht, np.diag(1-np.power(a_s[:,-1], 2))) 
		
		grad_W = np.matmul(grad_at.T, np.expand_dims(h_s[:,-2],1).T)
		grad_U = np.matmul(grad_at.T, np.expand_dims(input[:,-1],1).T)
		grad_b = grad_at.T
		for t in range(t_length-2,-1,-1):
			grad_ot = np.expand_dims(np.subtract(pred[:,t], output[:,t]), 1)
			grad_V += np.matmul(grad_ot, np.expand_dims(h_s[:,t+1],1).T)
			grad_c += grad_ot

			grad_ht = np.matmul(grad_ot.T,self.RNN["V"]) + np.matmul(grad_at, self.RNN["W"])
			grad_at = np.matmul(grad_ht, np.diag(1-np.power(a_s[:,t], 2)))

			grad_W += np.matmul(grad_at.T, np.expand_dims(h_s[:,t],1).T)
			grad_U += np.matmul(grad_at.T, np.expand_dims(input[:,t],1).T)
			grad_b += grad_at.T


		gradients = [grad_U, grad_W, grad_V, grad_b, grad_c]
		for index, grad in enumerate(gradients):
			gradients[index] = np.maximum(-5, np.minimum(grad, 5))



		return gradients

	def computeGradsNum(self,input, output, h_for_derivation=1e-4):
		all_grads_num = []
		h = np.zeros((self.m, 1))

		for layer_index, layer_name in enumerate(self.RNN):
			layer = self.RNN[layer_name]
			grad_elem = np.zeros(layer.shape)

			for i in range(layer.shape[0]):
				for j in range(layer.shape[1]):

					layer_try = np.copy(layer)
					layer_try[i, j] -= h_for_derivation
					all_weights_try = self.RNN.copy()
					all_weights_try[layer_name] = layer_try
					_,_,pred_try = self.forward(input, h, weigth_to_consider=all_weights_try)

					c1 = self.computeLoss(output, pred_try)

					layer_try = np.copy(layer)
					layer_try[i, j] += h_for_derivation
					all_weights_try = self.RNN.copy()
					all_weights_try[layer_name] = layer_try
					_,_,pred_try = self.forward(input, h, weigth_to_consider=all_weights_try)

					c2 = self.computeLoss(output, pred_try)

					grad_elem[i, j] = (c2-c1) / (2*h_for_derivation)

			all_grads_num.append(grad_elem)

		return all_grads_num


	def backpropagate(self, input, output, a_s, h_s, pred):
		grads = self.computeGrads(input, output, a_s, h_s, pred)
		for layer_id, layer_name in enumerate(self.RNN):
			self.AdaGrad_m[layer_name] += np.square(grads[layer_id])
			self.RNN[layer_name] -= self.eta*np.multiply(np.power(self.AdaGrad_m[layer_name] + epslion, -1/2), grads[layer_id])

			# # Update ada-grads
			# elem = grads[layer_id] ** 2
			# weight_elem = self.eta * grads[layer_id] / np.sqrt(self.AdaGrad_m[layer_name] + epslion)
			# a = 2



