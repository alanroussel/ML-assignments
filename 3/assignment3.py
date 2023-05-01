import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')

from load import load 
from utils import softmax, ReLU

#load dataset 
class Dataset:
	def __init__(self, mode):
		self.mode = mode
		print('dataset mode = {}'.format(mode))
		print("loading dataset ...")
		if(self.mode=="debug"):
			self.training = load('dataset/data_batch_1')
			self.validation = load('dataset/data_batch_2')
		else:
			dataset = [load('dataset/data_batch_1'), load('dataset/data_batch_2'), load('dataset/data_batch_3'), load('dataset/data_batch_4'), load('dataset/data_batch_5')]
			training = 0 #empty, to be concatenate
			validation = 0 #empty, to be concatenate
			print("splitting training and validation ...")
			for d_id, d in enumerate(dataset):
				print(f'\t{d_id+1}/5')
				perm = np.random.permutation(10000) 
				if training == 0:
					training = [d[0][perm[0:9000]], d[1][:,perm[0:9000]]]
					validation = [d[0][perm[9000:]], d[1][:,perm[9000:]]]
				else:
					training[0] = np.concatenate([training[0], d[0][perm[0:9000]]])
					training[1] = np.concatenate([training[1], d[1][:,perm[0:9000]]], 1)
					validation[0] = np.concatenate([validation[0], d[0][perm[9000:]]])
					validation[1] = np.concatenate([validation[1], d[1][:,perm[9000:]]], 1)
			self.training = training
			self.validation = validation
		self.test = load('dataset/test_batch')
		print(f'training dataset of size {self.training[0].shape[0]} \nvalidation dataset of size {self.validation[0].shape[0]}')
		print(f'test dataset of size {self.test[0].shape[0]}')
				
class Model:
	def __init__(self,m, lamda, batch_size, k, size_of_dataset):
		K = 10 #number of classes for the classification
		d = 1024*3 # dimension of the input
		# m number of neurons in the hidden layer

		ns = k*(size_of_dataset/batch_size) # 900 for 45000, 200 for 10000
		# 1 "cycle" means one up --> and one down <--. 
		# if k of ns = 2, 1 cycle means 4 epochs so 2 cycle means 8 epochs
		# if k of ns = 5, 1 cycle means 10 epochs 
		# if k of ns = 8, 1 cycle means 16 epochs so 2 cycle means 32 epochs

		self.layers = [
			{ "weight":np.random.normal(0,1/np.sqrt(d), (m, d)), "bias": np.random.normal(0, 0, (m, 1))},
			{ "weight":np.random.normal(0,1/np.sqrt(m), (K, m)), "bias": np.random.normal(0, 0, (K, 1))}
		]
		self.lamda = lamda
		self.eta_cycle_params = {
			"eta_min":1e-5, "eta_max":1e-1, "eta":1e-5, "up":True, "diff":((1e-1) - (1e-5))/ns
		}
	
	def eta_cycle(self):
		eta = self.eta_cycle_params["eta"]
		diff = self.eta_cycle_params["diff"]
		if(self.eta_cycle_params["up"]):
			eta += diff 
			if eta > self.eta_cycle_params["eta_max"]:
				self.eta_cycle_params["up"] = False
		else:
			eta -= diff
			if eta < self.eta_cycle_params["eta_min"]:
				self.eta_cycle_params["up"] = True
		if eta == 0:
			a  = 2
		self.eta_cycle_params["eta"] = eta
		
		return eta
		
	def evaluate(self, X):
		X = X.T
		
		H = ReLU(np.add(np.matmul(self.layers[0]["weight"], X), self.layers[0]["bias"])) 
		P = softmax(np.add(np.matmul(self.layers[1]["weight"], H), self.layers[1]["bias"]))
		
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
		cost = loss + self.lamda*weight_decay_sum 

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

		grad_W2 = np.matmul(G,H.T)/X.shape[0] + 2*self.lamda*self.layers[1]["weight"]
		grad_b2 = np.average(G,1).reshape(grad_b2.shape)

		G = np.matmul(self.layers[1]["weight"].T, G)
		G = np.multiply(G, np.where(H>0,1,0))

		grad_W1 = np.matmul(G, X)/X.shape[0] + 2*self.lamda*self.layers[0]["weight"]
		grad_b1 = np.average(G, 1).reshape(grad_b1.shape)
		return [grad_W1, grad_b1, grad_W2, grad_b2]
	
	def backpropagate(self, X, Y, H, P):
		eta = self.eta_cycle()
		[grad_W1, grad_b1, grad_W2, grad_b2] = self.computeGradsAnalytical(X,Y,H,P)
		self.layers[0]["weight"] -= eta*grad_W1
		self.layers[0]["bias"] -= eta*grad_b1
		self.layers[1]["weight"] -= eta*grad_W2
		self.layers[1]["bias"] -= eta*grad_b2

class Optimizer:
	def __init__(self,lamda, k, n_epochs, batch_size, dataset, m=50):
		self.model = Model(m, lamda, batch_size, k, dataset.training[0].shape[0])
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.str = f'lambda = {lamda}, n_epochs = {n_epochs}, batch_size = {batch_size}, m={m}'
		self.dataset = dataset
	
	def plot(self, plot_metrics, n_epochs):
		plt.subplot(133)
		plt.title('accuracy')
		plt.plot(range(1,n_epochs+1), plot_metrics[:,2], label="training")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,5], label="validation")
		plt.legend()
	
		plt.subplot(131)
		plt.title('cost')
		plt.plot(range(1,n_epochs+1), plot_metrics[:,1], label="training")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,4], label="validation")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,6], ".", alpha=0.2) 
		plt.legend()
		
		plt.subplot(132)
		plt.title('loss')
		plt.plot(range(1,n_epochs+1), plot_metrics[:,0], label="training")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,3], label="validation")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,6], ".", alpha=0.2) 
		plt.legend()


		plt.suptitle(self.str)
		plt.show()
		
		# fig, ax = plt.subplots(1,10, figsize=(16, 8))
		# for j in range(10):
		# 	im  = self.model.W[j,:].reshape(32,32,3, order='F')
		# 	sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
		# 	sim = sim.transpose(1,0,2)
		# 	ax[j].imshow(sim, interpolation='nearest')
		# 	ax[j].axis('off')
		# plt.show()

	def compare_two_gradients_methods(self):
		print(self.str)
		X, Y = np.copy(self.dataset.training[0]), np.copy(self.dataset.training[1])
		n_of_data = X.shape[0]
		perm = np.random.permutation(n_of_data)
		X = X[perm]
		Y = Y[:,perm]
		for batch_index in range(int(n_of_data/self.batch_size)):
				batch_id_start = batch_index*self.batch_size
				batch_id_end = (batch_index+1)*self.batch_size

				X_batch = X[batch_id_start:batch_id_end, :]
				Y_batch = Y[:, batch_id_start:batch_id_end]

				H, Y_pred = self.model.evaluate(X_batch)

				print(f'computing error for a batch ... ')
				analytical  = self.model.computeGradsAnalytical(X_batch, Y_batch,H, Y_pred)
				numeric = self.model.computeGradsNum(X_batch, Y_batch,Y_pred, h=1e-5)
				layers_name_to_print = ["W1", "b1", "W2", "b2"]
				for i, name in enumerate(layers_name_to_print):
					error = np.max(np.abs(analytical[i]-numeric[i])) / np.maximum(1e-6,np.abs(np.max(analytical[i])) + np.abs(np.max(numeric[i])))
					print(f'\terror is {error} for {name}')


	def resolve_with_SDG(self, verbose=True, plot=True, mode="training"):
		X, Y = np.copy(self.dataset.training[0]), np.copy(self.dataset.training[1])
		n_of_data = X.shape[0]
		X_v, Y_v = self.dataset.validation[0], self.dataset.validation[1]
		plot_metrics = np.zeros((self.n_epochs, 7))
		for epoch in range(self.n_epochs):
			perm = np.random.permutation(n_of_data)
			X = X[perm]
			Y = Y[:,perm]
			for batch_index in range(int(n_of_data/self.batch_size)):
				batch_id_start = batch_index*self.batch_size
				batch_id_end = (batch_index+1)*self.batch_size

				X_batch = X[batch_id_start:batch_id_end, :]
				Y_batch = Y[:, batch_id_start:batch_id_end]

				H, Y_pred = self.model.evaluate(X_batch)

				self.model.backpropagate(X_batch, Y_batch,H, Y_pred)

			# at each epoch
			if(not (mode=="tuning")):
				# if mode is training or debugging, we want to see the evolution of train and validation cost etc
				_, Ptraining = self.model.evaluate(X)
				_, Pvalidation = self.model.evaluate(X_v)
				
				t_loss, t_cost, t_accuracy= self.model.computeCost(Y, Ptraining)
				v_loss, v_cost, v_accuracy= self.model.computeCost(Y_v, Pvalidation)
				plot_metrics[epoch] = np.array([t_loss, t_cost, t_accuracy, v_loss, v_cost, v_accuracy, self.model.eta_cycle_params["eta"]])
				if(verbose):
					print('-- at end of epoch #{} --'.format(epoch))
					print(f'\t train loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
					print(f'\t valid loss/cost/accuracy = {v_loss:.5g} / {v_cost:.5g} / {v_accuracy:.3g}')
			
		if(mode=="training" or mode=="debug"):
			_, Ptest = self.model.evaluate(self.dataset.test[0])
			t_loss,t_cost, t_accuracy = self.model.computeCost(self.dataset.test[1], Ptest)
			print(f'\n\t test loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
			if(plot):
				self.plot(plot_metrics, self.n_epochs)
			return t_loss,t_cost, t_accuracy
		
		elif(mode=="tuning"):
			_, Pvalidation = self.model.evaluate(X_v)
			v_loss, v_cost, v_accuracy= self.model.computeCost(Y_v, Pvalidation)
			print(f'\n\t validation loss/cost/accuracy = {v_loss:.5g} / {v_cost:.5g} / {v_accuracy:.3g}')
			return v_loss, v_cost, v_accuracy

		


# mode can be debug, tuning or training  
mode = "debug"
dataset = Dataset(mode)

if mode=="debug":
	# debug gradients
	optimizer = Optimizer(lamda=0, k=5, n_epochs=5, batch_size=10, dataset=dataset, m=15)
	optimizer.compare_two_gradients_methods()

	#debug cyclical learning rate 
	# optimizer = Optimizer(lamda = 1e-2, k=5, n_epochs = 10, batch_size=100, dataset=dataset)
	## optimizer = Optimizer(lamda = 1e-2, k=8, n_epochs = 48, batch_size=100, dataset=dataset)
	# optimizer.resolve_with_SDG(plot=True, verbose=True, mode=mode)

if mode == "tuning":
	def broad_search():
		print("beginning of coarse search")
		lambdas = np.logspace(-5,-1,9)
		coarse_search = np.zeros((lambdas.shape[0], 3))
		for l_id, l in enumerate(lambdas):
			print(f'\nlamda = {l:.5g}')
			cost, accuracy = [], []
			for i in range(10):
				optimizer = Optimizer(lamda=l, k=2, n_epochs=8, batch_size=100, dataset=dataset)
				_, c, a = optimizer.resolve_with_SDG(plot=False, verbose=False, mode=mode)
				cost.append(c)
				accuracy.append(a)
			coarse_search[l_id] = np.array([l, np.average(cost), np.average(accuracy)])
		coarse_search[:,2] = coarse_search[:,2]*100
		print("end of coarse search")
		np.savetxt("coarse_search.txt",coarse_search,delimiter=',', fmt='%f')
	# broad_search()

	def narrow_search():
		print("beginning of narrow search")
		lambdas = np.logspace(-4,-2,11)
		narrow_search = np.zeros((lambdas.shape[0], 3))
		for l_id, l in enumerate(lambdas):
			print(f'\nlamda = {l:.5g}')
			accuracy = []
			cost = []
			for i in range(10):
				optimizer = Optimizer(lamda=l, k=2, n_epochs=8, batch_size=100, dataset=dataset)
				loss, c, a = optimizer.resolve_with_SDG(plot=False, verbose=False, mode=mode)
				accuracy.append(a)
				cost.append(c)
			narrow_search[l_id] = np.array([l, np.average(cost), np.average(accuracy)])
		narrow_search[:,2] = narrow_search[:,2]*100
		print("end of narrow search")
		np.savetxt("narrow_search.txt",narrow_search,delimiter=',', fmt='%f')
	# narrow_search()

if mode == "training":
	# best parameters are lambda = e-3
	optimizer = Optimizer(lamda=1e-3, k=8, n_epochs=32, batch_size=100, dataset=dataset)
	loss, c, a = optimizer.resolve_with_SDG(plot=True, verbose=True, mode=mode)
	#overfiiting parameters : 0:100 of training





