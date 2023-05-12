import matplotlib.pyplot as plt
import numpy as np

from datasetLoader import DatasetLoader
from optimizer import Optimizer

# MODE CONFIG - debug, training, or tuning ? 
mode = "training"
dataset = DatasetLoader(mode) 

if mode=="debug":
	# debug gradients
	
	# simple training
	# optimizer = Optimizer(weight_decay_parameter=0, lr_cycle_magnitude=5, n_epochs=20, batch_size=15, dataset=dataset, hidden_layers_structure=[12,13],batch_normalization=False)
	# optimizer.compare_two_gradients_methods()

	
	optimizer = Optimizer(weight_decay_parameter=0, lr_cycle_magnitude=5, n_epochs=20, batch_size=15, dataset=dataset, hidden_layers_structure=[12,13,14],batch_normalization=False)
	optimizer.compare_two_gradients_methods()
	optimizer = Optimizer(weight_decay_parameter=0, lr_cycle_magnitude=5, n_epochs=20, batch_size=15, dataset=dataset, hidden_layers_structure=[12,13],batch_normalization=True)
	optimizer.compare_two_gradients_methods()
	

	# optimizer = Optimizer(weight_decay_parameter=0, lr_cycle_magnitude=5, n_epochs=20, batch_size=15, dataset=dataset, hidden_layers_structure=[12,13],batch_normalization=True)
	# optimizer.resolve_with_SDG(plot=True, verbose=True, mode=mode)

	#debug cyclical learning rate 
	# optimizer = Optimizer(weight_decay_parameter = 1e-2, lr_cycle_magnitude=5, n_epochs = 10, batch_size=100, dataset=dataset)
	## optimizer = Optimizer(weight_decay_parameter = 1e-2, lr_cycle_magnitude=8, n_epochs = 48, batch_size=100, dataset=dataset)
	

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

	# first training on 50,50
	optimizer = Optimizer(weight_decay_parameter=5e-3, lr_cycle_magnitude=5, n_epochs=20, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 50], batch_normalization=False, verbose=True)
	loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)

	optimizer = Optimizer(weight_decay_parameter=5e-3, lr_cycle_magnitude=5, n_epochs=20, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 50], batch_normalization=True, verbose=True)
	loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)


	#first training on 9 layers 
	# optimizer = Optimizer(weight_decay_parameter=5e-3, lr_cycle_magnitude=5, n_epochs=20, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 30, 20, 20, 10, 10, 10, 10], batch_normalization=False, verbose=True)
	# loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)

	# optimizer = Optimizer(weight_decay_parameter=5e-3, lr_cycle_magnitude=5, n_epochs=20, batch_size=100, dataset=dataset,hidden_layers_structure=[50, 30, 20, 20, 10, 10, 10, 10], batch_normalization=True, verbose=True)
	# loss, c, a = optimizer.resolve_with_SDG(plot=True, mode=mode)


	





