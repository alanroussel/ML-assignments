
import numpy as np

def show_weights(W):
	""" Display the W W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1,10, figsize=(16, 8))
	for j in range(10):
		im  = W[j,:].reshape(32,32,3, order='F')
		sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
		sim = sim.transpose(1,0,2)
		ax[j].imshow(sim, interpolation='nearest')
		ax[j].axis('off')
	plt.show()
	
