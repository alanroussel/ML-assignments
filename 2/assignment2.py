import matplotlib.pyplot as plt
import numpy as np
from load import load 
from utils import softmax, ReLU

#load dataset 
print("loading full dataset ...")
dataset = [load('dataset/data_batch_1'), load('dataset/data_batch_2'), load('dataset/data_batch_3'), load('dataset/data_batch_4'), load('dataset/data_batch_5')]
training = 0
validation = 0
print("\tsplit training and validation :")
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
test = load('dataset/test_batch')
a = 5
class Model:
	def __init__(self,lamda, batch_size):
		K = 10 #number of classes for the classification
		d = 1024*3 # dimension of the input
		m = 50 # number of neurons in the hidden layer

		ns = 2*(training[0].shape[0]/batch_size) # 900 for 45000, 200 for 10000
		# 1 "cycle" means one up --> and one down <--. 
		# 1 cycle = 4 epoch, 2 cycle = 8 epochs

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

	def computeGradsNumCorrection(self, X, Y,P,h=1e-6):
		""" Converted from matlab code """
		grad_W = np.zeros(self.W.shape)
		grad_b = np.zeros(self.b.shape)

		c, _, _ = self.computeCost(Y, P, andAccuracy=False)

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

# model = Model(1, 0.001)
# X, Y = training[0][0:3,:], training[1][:,0:3]
# H,P = model.evaluate(X)
# model.computeGradsAnalytical(X,Y, H, P)

class Optimizer:
	def __init__(self,lamda, n_epochs, batch_size):
		self.model = Model(lamda, batch_size)
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.str = f'lambda = {lamda}, n_epochs = {n_epochs}, batch_size = {batch_size} '
	
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
		# plt.plot(range(1,n_epochs+1), plot_metrics[:,6], ".", alpha=0.2) #only for debug purpose
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

	def resolve_with_SDG(self, verbose=True, plot=True):
		X, Y = training[0], training[1]
		n_of_data = X.shape[0]
		X_v, Y_v = validation[0], validation[1]
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
				# [a_grad_W, a_grad_b] = self.model.computeGradsAnalytical(X_batch, Y_batch, Y_pred)
				# [n_grad_W, n_grad_b] = self.model.computeGradsNumCorrection(X_batch, Y_batch, Y_pred)
				# error = np.max(np.abs(a_grad_W-n_grad_W)) / np.maximum(1e-6,np.abs(np.max(a_grad_W)) + np.abs(np.max(n_grad_W)))
				# print(error)
			_, Ptraining = self.model.evaluate(X)
			_, Pvalidation = self.model.evaluate(X_v)
			t_loss, t_cost, t_accuracy= self.model.computeCost(Y, Ptraining)
			v_loss, v_cost, v_accuracy= self.model.computeCost(Y_v, Pvalidation)

			if(verbose):
				print('-- at end of epoch #{} --'.format(epoch))
				print(f'\t train loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
				print(f'\t valid loss/cost/accuracy = {v_loss:.5g} / {v_cost:.5g} / {v_accuracy:.3g}')
			plot_metrics[epoch] = np.array([t_loss, t_cost, t_accuracy, v_loss, v_cost, v_accuracy, self.model.eta_cycle_params["eta"]])
		
		_, Ptest = self.model.evaluate(test[0])
		t_loss,t_cost, t_accuracy = self.model.computeCost(test[1], Ptest)
		print(f'\tFinal results \n\t test loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
		if(plot):
			self.plot(plot_metrics, self.n_epochs)
		return t_loss,t_cost, t_accuracy


# Optimizer = Optimizer(K,d,lamda=0, n_epochs=40, batch_size=100, eta=.1)
# Optimizer = Optimizer(K,d,lamda=0, n_epochs=40, batch_size=100, eta=.001)
# Optimizer = Optimizer(K,d,lamda=.1, n_epochs=40, batch_size=100, eta=.001)

#first broad search
def broad_search():
	print("beginning of coarse search")
	lambdas = np.logspace(-5,-1,9)
	coarse_search = np.zeros((lambdas.shape[0], 2))
	for l_id, l in enumerate(lambdas):
		print(f'\nlamda = {l:.5g}')
		accuracy = []
		for i in range(5):
			optimizer = Optimizer(lamda=l, n_epochs=8, batch_size=100)
			_, _, a = optimizer.resolve_with_SDG(plot=False, verbose=False)
			accuracy.append(a)
		coarse_search[l_id] = np.array([l, np.average(accuracy)])
	coarse_search[:,1] = coarse_search[:,1]*100
	print("end of coarse search")
	np.savetxt("coarse_search.txt",coarse_search,delimiter=',', fmt='%f')
broad_search()

# def narrow_search():
# 	print("beginning of narrow search")
# 	lambdas = np.logspace(-5,-1,9)
# 	narrow_search = np.zeros((lambdas.shape[0], 2))
# 	for l_id, l in enumerate(lambdas):
# 		print(f'\nlamda = {l:.5g}')
# 		optimizer = Optimizer(lamda=l, n_epochs=8, batch_size=100)
# 		loss, cost, accuracy = optimizer.resolve_with_SDG(plot=False, verbose=False)
# 		narrow_search[l_id] = np.array([l, accuracy])
# 	narrow_search[:,1] = narrow_search[:,1]*100
# 	print("end of narrow search")
# 	np.savetxt("narrow_search.txt",narrow_search,delimiter=',', fmt='%f')

# narrow_search()


#overfiiting parameters : 0:100 of training





