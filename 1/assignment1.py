import matplotlib.pyplot as plt
import numpy as np
import random
from load import load 
from utils import softmax



# # init model
# W = np.random.normal(0, 0.01, (K, d))
# b = np.random.normal(0, 0.01, (K, 1))

#load dataset
training = load('dataset/data_batch_1')
validation = load('dataset/data_batch_2')
test = load('dataset/test_batch')

class Model:
	def __init__(self, K, d, lamda, eta):
		self.W = np.random.normal(0, 0.01, (K, d))
		self.b = np.random.normal(0, 0.01, (K, 1))
		self.lamda = lamda
		self.eta = eta 
	
	def evaluate(self, X):
		Ypred = np.add(np.matmul(self.W, X.T), self.b)
		Ypred_soft_max = softmax(Ypred)
		return Ypred_soft_max

	def computeCost(self,Y,Ypred,andAccuracy=True):
		'''
		input: dataloader, weigth, bias, weight decay parameter, and wheter to compute or not the accuracy
		returns the loss, cost(+weight decay) and accuracy
		'''
		batch_size = Y.shape[1]
		
		loss, cost, accuracy = 0,0, 0
		for im_idx in range(batch_size):
			loss -= np.matmul(Y[:,im_idx], np.log(Ypred[:,im_idx]))
		loss /= batch_size
		cost = loss + self.lamda*np.sum(np.square(self.W)) # weight decay 

		if(andAccuracy):
			accuracy = np.sum(np.argmax(Ypred, 0) == np.argmax(Y, 0))/batch_size
		return loss, cost, accuracy

	def computeGradsNumCorrection(self, X, Y,Ypred,h=1e-6):
		""" Converted from matlab code """
		grad_W = np.zeros(self.W.shape)
		grad_b = np.zeros(self.b.shape)

		c, _, _ = self.computeCost(Y, Ypred, andAccuracy=False)

		for i in range(len(self.b)):
			b_try = np.array(self.b)
			b_try[i] += h
			Y_pred = softmax(np.add(np.matmul(self.W, X.T), b_try))
			c2, _, _ = self.computeCost(Y, Y_pred, andAccuracy=False)
			grad_b[i] = (c2-c) / h

		for i in range(self.W.shape[0]):
			for j in range(self.W.shape[1]):
				W_try = np.array(self.W)
				W_try[i,j] += h
				Y_pred = softmax(np.add(np.matmul(W_try, X.T), self.b))
				c2, _, _ = self.computeCost(Y, Y_pred, andAccuracy=False)
				grad_W[i,j] = (c2-c) / h

		return [grad_W, grad_b]

	def computeGradsAnalytical(self,X,Y,Ypred):
		""" Converted from matlab code """
		grad_W = np.zeros(self.W.shape)
		grad_b = np.zeros(self.b.shape)
		G = -(Y-Ypred)
		grad_W = np.matmul(G, X)/X.shape[0] + 2*self.lamda*self.W
		grad_b = np.average(G, 1).reshape(self.b.shape)
		return [grad_W, grad_b]
	
	def backpropagate(self, X, Y, Ypred):
		[grad_W, grad_b] = self.computeGradsAnalytical(X,Y,Ypred)
		self.W -= self.eta*grad_W
		self.b -= self.eta*grad_b

class Resolve:
	def __init__(self, K, d, lamda, n_epochs, n_batch, eta):
		self.model = Model(K, d, lamda, eta)
		self.n_epochs = n_epochs
		self.n_batch = n_batch
		self.str = f'lambda = {lamda}, n_epochs = {n_epochs}, n_batch = {n_batch}, eta = {eta}  '
	
	def plot(self, plot_metrics, n_epochs):
		plt.subplot(121)
		plt.title('accuracy')
		plt.plot(range(1,n_epochs+1), plot_metrics[:,2], label="training")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,5], label="validation")
		plt.legend()
	
		plt.subplot(122)
		plt.title('cost')
		plt.plot(range(1,n_epochs+1), plot_metrics[:,0], label="lost training")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,3], label="lost validation")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,1], label="cost training")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,4], label="cost validation")
		plt.legend()
		plt.suptitle(self.str)
		plt.show()
		
		fig, ax = plt.subplots(1,10, figsize=(16, 8))
		for j in range(10):
			im  = self.model.W[j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[j].imshow(sim, interpolation='nearest')
			ax[j].axis('off')
		plt.show()

	def resolve_with_SDG(self, verbose=True, plot=True):
		X, Y = training[0], training[1]
		n_of_data = X.shape[0]
		X_v, Y_v = validation[0], validation[1]
		plot_metrics = np.zeros((self.n_epochs, 6))
		for epoch in range(self.n_epochs):
			perm = np.random.permutation(n_of_data)
			X = X[perm]
			Y = Y[:,perm]
			for batch_index in range(int(n_of_data/self.n_batch)):
				batch_id_start = batch_index*self.n_batch
				batch_id_end = (batch_index+1)*self.n_batch

				X_batch = X[batch_id_start:batch_id_end, :]
				Y_batch = Y[:, batch_id_start:batch_id_end]

				Y_pred = self.model.evaluate(X_batch)

				self.model.backpropagate(X_batch, Y_batch, Y_pred)
				[a_grad_W, a_grad_b] = self.model.computeGradsAnalytical(X_batch, Y_batch, Y_pred)
				[n_grad_W, n_grad_b] = self.model.computeGradsNumCorrection(X_batch, Y_batch, Y_pred)
				error = np.max(np.abs(a_grad_W-n_grad_W)) / np.maximum(1e-6,np.abs(np.max(a_grad_W)) + np.abs(np.max(n_grad_W)))
				print(error)
		
			Ypredtraining = self.model.evaluate(X)
			Ypredvalidation = self.model.evaluate(X_v)
			t_loss, t_cost, t_accuracy= self.model.computeCost(Y, Ypredtraining)
			v_loss, v_cost, v_accuracy= self.model.computeCost(Y_v, Ypredvalidation)

			if(verbose):
				print('-- at end of epoch #{} --'.format(epoch))
				print(f'\t train loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
				print(f'\t valid loss/cost/accuracy = {v_loss:.5g} / {v_cost:.5g} / {v_accuracy:.3g}')
			plot_metrics[epoch] = np.array([t_loss, t_cost, t_accuracy, v_loss, v_cost, v_accuracy])
		
		
		print('\n\nFinal results : ')
		Ypredtest = self.model.evaluate(test[0])
		t_loss,t_cost, t_accuracy = self.model.computeCost(test[1], Ypredtest)
		print(f'\t test loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
		if(plot):
			self.plot(plot_metrics, self.n_epochs)


#hyperparameters of the dataset
K = 10 # numbers of class
d = 1024*3 # numbers of pixels (RGB)



# resolve = Resolve(K,d,lamda=0, n_epochs=40, n_batch=100, eta=.1)
# resolve = Resolve(K,d,lamda=0, n_epochs=40, n_batch=100, eta=.001)
# resolve = Resolve(K,d,lamda=.1, n_epochs=40, n_batch=100, eta=.001)
resolve = Resolve(K,d,lamda=1, n_epochs=40, n_batch=100, eta=.001)



resolve.resolve_with_SDG(plot=True)





