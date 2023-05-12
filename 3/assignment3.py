import numpy as np 
import matplotlib.pyplot as plt

# MODE CONFIG - debug, training, or tuning ? 
mode = "training"



# dataset class
def normalize_images(X):
	btw_0_and_1 = np.zeros(X.shape)
	for im_dx, im in enumerate(X):
		btw_0_and_1[im_dx] = im / 255
	
	mean = np.mean(btw_0_and_1, 0)
	standart_deviation = np.std(btw_0_and_1, 0)
	return np.divide(np.subtract(btw_0_and_1, mean), standart_deviation)

def one_hot(y, K):
	classes = np.zeros((len(y), K))
	classes[np.arange(len(y)), y] = 1
	return classes.T

def load(file, K=10):
	'''
	file is path
	K is number of class, here 10
	'''
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	X = normalize_images(dict[b'data'])
	y = np.array(dict[b'labels'])
	Y = one_hot(y, K)
	return [X, Y]

class DatasetLoader:
	def __init__(self, mode):
		self.mode = mode
		print(f'dataset \n{mode} mode')
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
		print(f'training/validation/test : {self.training[0].shape[0]}/{self.validation[0].shape[0]}/{self.test[0].shape[0]} \n')
		
	
# eta cycle class

class EtaCycle:
	def __init__(self, eta_min, eta_max, ns):
		self.eta_min = eta_min
		self.eta_max = eta_max
		self.eta = eta_min # current value of eta
		self.direction = True # True is up
		self.diff = (eta_max-eta_min)/ns
	
	def next(self):
		if(self.direction):
			self.eta += self.diff 
			if self.eta > self.eta_max:
				self.direction = False
		else:
			self.eta -= self.diff
			if self.eta < self.eta_min:
				self.direction = True

	def getEta(self):
		return self.eta


# batch normalization class

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
		
# model class

epslion = 1e-15
alpha = 0.9
def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def ReLU(x):
	""" Standard definition of relu """
	return np.maximum(x, 0) #to be checked

class Model:
	def __init__(self,hidden_layers_structure, weight_decay_parameter, batch_size, lr_cycle_magnitude, size_of_dataset, batch_normalization, verbose):
		"""
		hidden_layers_structure : array containing size of each hidden layer. ex [50] or [15,15]
		lambda : weight decay parameter
		batch size : number of point to take at each batch
		lr_cycle_magnitude : parameter for the cyclical learning rate
		size of dataset : size of the training dataset
		batch_normalization : do we include batch normalization ?
		"""

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
			shape_weight = (layers_sizes[l+1], layers_sizes[l])
			shape_activation = (layers_sizes[l+1],1)
			self.layers.append(
				{ 
					"weight":np.random.normal(0,1/np.sqrt(layers_sizes[l]),shape_weight), 
					"bias": np.random.normal(0,1/np.sqrt(layers_sizes[l]),shape_activation),
					"scale":np.ones(shape_activation),
					"shift":np.ones(shape_activation),
					"batch_normalization":BatchNormalization(shape_activation)
			})

		#remove useless info from last layer, and from all
		self.layers[-1].pop("scale")
		self.layers[-1].pop("shift")
		self.batch_normalization = batch_normalization

		self.weight_decay_parameter = weight_decay_parameter
		self.etaCycle = EtaCycle(1e-5, 1e-1, ns)
		if(verbose):
			print(f'model \n size of layers : {layers_sizes} \n weight decay parameter : {weight_decay_parameter} \n batch normalization : {batch_normalization} \n')
		self.plot_title = f'layers : {layers_sizes}, weight decay parameter : {weight_decay_parameter} batch normalization : {batch_normalization}'
	
	def evaluate(self, X, step):
		'''
		X in the input, the flatten image
		step is training or testing time 
		During training, we computed the mean and variance of the batch, and keep it in order to have averages
		During testing, the un-normalized scores are normalized by known pre-computed means and variances that have been estimated during training
		'''
		X = X.T
		metrics = [(0,0,X)]
		for layer in self.layers[0:-1]:
			S = np.add(np.matmul(layer["weight"], X), layer["bias"])
			
			S_to_store, S_hat = None, None #init 
			if(self.batch_normalization):
				if step=="training_time":
					layer["batch_normalization"].compute(S)
				mu, var = layer["batch_normalization"].getMuAndVar(step)	
				S_hat = np.divide(S-mu, np.sqrt(var + epslion))
				S_to_store = np.copy(S)
				S = np.add(np.multiply(S_hat, layer["scale"]), layer["shift"])
				
			X = ReLU(S)

			metrics.append((S_to_store, S_hat, X))

		P = softmax(np.add(np.matmul(self.layers[-1]["weight"], X), self.layers[-1]["bias"]))

		
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

	def computeGradsNum(self, X, Y,P,h):
		""" Converted from matlab code """
		grad_W1 = np.zeros(self.layers[0]["weight"].shape)
		grad_b1 = np.zeros(self.layers[0]["bias"].shape)

		X = X.T

		c, _, _ = self.computeCost(Y, P, andAccuracy=False)
		if(self.batch_normalization):
			for i in range(grad_W1.shape[0]):
				for j in range(grad_W1.shape[1]):
					W_try = np.array(self.layers[0]["weight"])
					W_try[i,j] += h
					Xtry = np.copy(X)

					Stry = np.add(np.matmul(W_try, Xtry), self.layers[0]["bias"])
					mean, var = np.average(Stry, 1), np.var(Stry, 1)
					Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1) + epslion ))
					Stry_wave = np.add(np.multiply(Stry_hat, self.layers[0]["scale"]), self.layers[0]["shift"])
					Xtry = ReLU(Stry_wave)

					for layer in self.layers[1:-1]:
						Stry = np.add(np.matmul(layer["weight"], Xtry), layer["bias"])
						mean, var = np.average(Stry, 1), np.var(Stry, 1)
						Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)+ epslion))
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
				Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)+ epslion))
				Stry_wave = np.add(np.multiply(Stry_hat, self.layers[0]["scale"]), self.layers[0]["shift"])
				Xtry = ReLU(Stry_wave)

				for layer in self.layers[1:-1]:
					Stry = np.add(np.matmul(layer["weight"], Xtry), layer["bias"])
					mean, var = np.average(Stry, 1), np.var(Stry, 1)
					Stry_hat = np.divide(Stry-mean.reshape(mean.shape[0], 1), np.sqrt(var.reshape(var.shape[0], 1)+epslion))
					Stry_wave = np.add(np.multiply(Stry_hat, layer["scale"]), layer["shift"])
					Xtry = ReLU(Stry_wave)
				Ptry = softmax(np.add(np.matmul(self.layers[-1]["weight"], Xtry), self.layers[-1]["bias"]))
				c2, _, _ = self.computeCost(Y, Ptry, andAccuracy=False)
				grad_b1[i] = (c2-c) / h
			
		else:
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

	def computeGradsAnalytical(self,X,Y,P, metrics):
		N = X.shape[0] #batch size
		#unzip metrics
		Ss, S_hats, Xs = zip(*metrics)
		if(self.batch_normalization):
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
					
					# new implementation
					mu, var = layer["batch_normalization"].getMuAndVar("training_time")
					gamma1 = np.power(var+epslion, -0.5)
					gamma2 = np.power(var+epslion, -1.5)

					G1 = np.multiply(G, gamma1)
					G2 = np.multiply(G, gamma2)
					D = np.subtract(Ss[-1-layer_id], mu)
					c = np.sum(np.multiply(G2,D), 1)
					c = c.reshape(c.shape[0],1)
					G = np.subtract(np.subtract(G1, np.average(G1,1).reshape(G1.shape[0],1)), np.multiply(D,np.matmul(c,np.ones((1,N))))/N)

					grad_weight = np.matmul(G, Xs[-1-layer_id-1].T)/N + 2*self.weight_decay_parameter*layer["weight"]
					grad_bias = np.average(G,1).reshape(layer["bias"].shape)
					G = np.matmul(layer["weight"].T, G)
					G = np.multiply(G, np.where(Xs[-1-layer_id-1]>0,1,0))

				grads.append([grad_weight, grad_bias, grad_scale, grad_shift])
		
		else:
			grads = []
			G = -(Y-P)
			
			for layer_id, layer in enumerate(reversed(self.layers)):
				if layer_id != len(self.layers)-1:
					grad_weight = np.matmul(G,Xs[-1-layer_id].T)/N + 2*self.weight_decay_parameter*layer["weight"]
					grad_bias = np.average(G,1).reshape(layer["bias"].shape)
					G = np.matmul(layer["weight"].T, G)
					G = np.multiply(G, np.where(Xs[-1-layer_id]>0,1,0))
				else:
					#computing gradients for the first layer
					grad_weight = np.matmul(G,X)/N + 2*self.weight_decay_parameter*layer["weight"]
					grad_bias = np.average(G,1).reshape(layer["bias"].shape)
				grads.append([grad_weight, grad_bias])
				
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
			

# optimizer class

class Optimizer:
	def __init__(self,weight_decay_parameter, lr_cycle_magnitude, n_epochs, batch_size, dataset, hidden_layers_structure, batch_normalization, verbose):
		self.model = Model(hidden_layers_structure, weight_decay_parameter, batch_size, lr_cycle_magnitude, dataset.training[0].shape[0], batch_normalization, verbose)
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.dataset = dataset
		self.verbose = verbose
		if(verbose):
			print(f'optimizer \n n_epochs : {n_epochs} \n batch_size : {batch_size} \n')
	
	def plot(self, plot_metrics, n_epochs):
		plt.subplot(122)
		plt.title('accuracy')
		plt.plot(range(1,n_epochs+1), plot_metrics[:,2], label="training")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,5], label="validation")
		plt.legend()
	
		
		plt.subplot(121)
		plt.title('loss')
		plt.plot(range(1,n_epochs+1), plot_metrics[:,0], label="training")
		plt.plot(range(1,n_epochs+1), plot_metrics[:,3], label="validation")
		plt.legend()


		# plt.suptitle(self.model.plot_title)
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
		X, Y = np.copy(self.dataset.training[0]), np.copy(self.dataset.training[1])
		n_of_data = X.shape[0]
		perm = np.random.permutation(n_of_data)
		X = X[perm]
		Y = Y[:,perm]
		for batch_index in range(int(n_of_data/self.batch_size)):
				if(batch_index==3):
					break
				batch_id_start = batch_index*self.batch_size
				batch_id_end = (batch_index+1)*self.batch_size
				pass

				X_batch = X[batch_id_start:batch_id_end, :]
				Y_batch = Y[:, batch_id_start:batch_id_end]

				metrics, Y_pred = self.model.evaluate(X_batch, step="training_time")

				print(f'computing error for a batch ... ')
				analytical = self.model.computeGradsAnalytical(X_batch, Y_batch,Y_pred, metrics)
				numeric = self.model.computeGradsNum(X_batch, Y_batch,Y_pred, 1e-6)
				layers_name_to_print = ["W1", "b1"]
				for i, name in enumerate(layers_name_to_print):
					error = np.max(np.abs(analytical[0][i]-numeric[i])) / np.maximum(1e-6,np.abs(np.max(analytical[0][i])) + np.abs(np.max(numeric[i])))
					print(f'\terror is {error} for {name}')

	def resolve_with_SDG(self, plot=True, mode="training"):
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

				metrics, Y_pred = self.model.evaluate(X_batch, step="training_time")
				self.model.backpropagate(X_batch, Y_batch,Y_pred, metrics)

			# at each epoch
			if(not (mode=="tuning")):
				# if mode is training or debugging, we want to see the evolution of train and validation cost etc
				_, Ptraining = self.model.evaluate(X, step="testing_time")
				_, Pvalidation = self.model.evaluate(X_v, step="testing_time")
				
				t_loss, t_cost, t_accuracy= self.model.computeCost(Y, Ptraining)
				v_loss, v_cost, v_accuracy= self.model.computeCost(Y_v, Pvalidation)
				plot_metrics[epoch] = np.array([t_loss, t_cost, t_accuracy, v_loss, v_cost, v_accuracy, self.model.etaCycle.eta])
				if(self.verbose):
					print('-- at end of epoch #{} --'.format(epoch))
					print(f'\t train loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
					print(f'\t valid loss/cost/accuracy = {v_loss:.5g} / {v_cost:.5g} / {v_accuracy:.3g}')
			
		if(mode=="training" or mode=="debug"):
			_, Ptest = self.model.evaluate(self.dataset.test[0],step="testing_time")
			t_loss,t_cost, t_accuracy = self.model.computeCost(self.dataset.test[1], Ptest)
			print(f'\n\t test loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
			if(plot):
				self.plot(plot_metrics, self.n_epochs)
			return t_loss,t_cost, t_accuracy
		
		elif(mode=="tuning"):
			_, Pvalidation = self.model.evaluate(X_v, step="testing_time")
			v_loss, v_cost, v_accuracy= self.model.computeCost(Y_v, Pvalidation)
			print(f'\n\t validation loss/cost/accuracy = {v_loss:.5g} / {v_cost:.5g} / {v_accuracy:.3g}')
			return v_loss, v_cost, v_accuracy




dataset = DatasetLoader(mode) 

if mode=="debug":
	
	optimizer = Optimizer(weight_decay_parameter=0, lr_cycle_magnitude=5, n_epochs=20, batch_size=15, dataset=dataset, hidden_layers_structure=[12,13,14],batch_normalization=False, verbose=True)
	optimizer.compare_two_gradients_methods()
	
	optimizer = Optimizer(weight_decay_parameter=0, lr_cycle_magnitude=5, n_epochs=20, batch_size=15, dataset=dataset, hidden_layers_structure=[12,13],batch_normalization=True, verbose=True)
	optimizer.compare_two_gradients_methods()
	

if mode == "tuning":
	print("beginning of coarse search")
	lambdas = np.logspace(-5,-1,9)
	coarse_search = np.zeros((lambdas.shape[0], 3))
	for l_id, l in enumerate(lambdas):
		print(f'\nweight_decay_parameter = {l:.5g}')
		cost, accuracy = [], []
		for i in range(3):
			optimizer = Optimizer(weight_decay_parameter=l, lr_cycle_magnitude=2, n_epochs=8, batch_size=100, dataset=dataset, hidden_layers_structure=[50, 50], batch_normalization=True, verbose=False)
			_, c, a = optimizer.resolve_with_SDG(plot=False, mode=mode)
			cost.append(c)
			accuracy.append(a)
		coarse_search[l_id] = np.array([l, np.average(cost), np.average(accuracy)])
	coarse_search[:,2] = coarse_search[:,2]*100
	print("end of coarse search")
	np.savetxt("coarse_search.txt",coarse_search,delimiter=',', fmt='%f')


if mode == "training":
	# best parameters are lambda = 3e-3
	optimizer = Optimizer(weight_decay_parameter=3e-3, lr_cycle_magnitude=5, n_epochs=30, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 50], batch_normalization=True, verbose=True)
	loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)

	# first training on 50,50
	# optimizer = Optimizer(weight_decay_parameter=5e-3, lr_cycle_magnitude=5, n_epochs=20, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 50], batch_normalization=False, verbose=True)
	# loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)

	# optimizer = Optimizer(weight_decay_parameter=5e-3, lr_cycle_magnitude=5, n_epochs=20, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 50], batch_normalization=True, verbose=True)
	# loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)


	#first training on 9 layers 
	# optimizer = Optimizer(weight_decay_parameter=5e-3, lr_cycle_magnitude=5, n_epochs=20, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 30, 20, 20, 10, 10, 10, 10], batch_normalization=False, verbose=True)
	# loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)

	# optimizer = Optimizer(weight_decay_parameter=5e-3, lr_cycle_magnitude=5, n_epochs=20, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 30, 20, 20, 10, 10, 10, 10], batch_normalization=True, verbose=True)
	# loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)

	


	








