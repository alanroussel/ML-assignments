import numpy as np
import matplotlib.pyplot as plt
from model import Model
class Optimizer:
	def __init__(self,weight_decay_parameter, lr_cycle_magnitude, n_epochs, batch_size, dataset, hidden_layers_structure, batch_normalization):
		self.model = Model(hidden_layers_structure, weight_decay_parameter, batch_size, lr_cycle_magnitude, dataset.training[0].shape[0], batch_normalization)
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.dataset = dataset
		print(f'optimizer \n n_epochs : {n_epochs} \n batch_size : {batch_size} \n')
	
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


		plt.suptitle(self.model.plot_title)
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
				if(verbose):
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

	