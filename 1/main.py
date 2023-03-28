import matplotlib.pyplot as plt
import numpy as np
import random
from load import load 
from utils import softmax, montage

#hyperparameters of the dataset
K = 10 # numbers of class
d = 1024*3 # numbers of pixels (RGB)
weigth_decay_lamdba = 0

# init model
W = np.random.normal(0, 0.01, (K, d))
b = np.random.normal(0, 0.01, (K, 1))

#load dataset
training = load('dataset/data_batch_1')
validation = load('dataset/data_batch_2')
test = load('dataset/test_batch')

def evaluateClassifier(X, W, b):
	Ypred = np.add(np.matmul(W, X.T), b)
	Ypred_soft_max = softmax(Ypred)
	return Ypred_soft_max
	
def computeCost(X, Y, W, b,lamda, andAccuracy=True):
    '''
    input: dataloader, weigth, bias, weight decay parameter, and wheter to compute or not the accuracy
    returns the loss, cost(+weight decay) and accuracy
    '''
    Ypred = evaluateClassifier(X,W,b)
    batch_size = len(X)
    
    loss, cost, accuracy = 0,0, 0
    for im_idx in range(batch_size):
        loss -= np.matmul(Y[:,im_idx], np.log(Ypred[:,im_idx]))
    loss /= batch_size
    cost = loss + lamda*np.sum(np.square(W)) # weight decay 

    if(andAccuracy):
	    accuracy = np.sum(np.argmax(Ypred, 0) == np.argmax(Y, 0))/batch_size
    return loss, cost, accuracy


# compute gradient
def computeGradsNumCorrection(X, Y, W, b, lamda, h=1e-6):
	""" Converted from matlab code """
	grad_W = np.zeros(W.shape)
	grad_b = np.zeros(b.shape)

	c, _, _ = computeCost(X, Y, W, b, lamda, andAccuracy=False)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2, _, _ = computeCost(X, Y, W, b_try, lamda, andAccuracy=False)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2, _, _ = computeCost(X, Y, W_try, b, lamda, andAccuracy=False)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def computeGradsAnalytical(X, Y, W, b, lamda):
	""" Converted from matlab code """
	grad_W = np.zeros(W.shape)
	grad_b = np.zeros(b.shape)
	Ypred = evaluateClassifier(X, W, b)
	G = -(Y-Ypred)
	grad_W = np.matmul(G, X)/X.shape[0] + 2*lamda*W
	grad_b = np.average(G, 1).reshape(b.shape)
	return [grad_W, grad_b]

def resolveWithSDG(W, b, lamda, n_epochs, n_batch, eta):
    X, Y = training[0], training[1]
    n_of_data = X.shape[0]
    X_v, Y_v = validation[0], validation[1]

    plot = np.zeros((n_epochs, 6))
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_of_data)
        X = X[perm]
        Y = Y[:,perm]
        for batch_index in range(int(n_of_data/n_batch)):
            batch_id_start = batch_index*n_batch
            batch_id_end = (batch_index+1)*n_batch

            X_batch = X[batch_id_start:batch_id_end, :]
            Y_batch = Y[:, batch_id_start:batch_id_end]
	    
            [n_grad_W, n_grad_b] = computeGradsAnalytical(X_batch, Y_batch, W, b, lamda)
	    
            # [a_grad_W, a_grad_b] = computeGradsNumCorrection(X_batch, Y_batch,W,b,lamda)

            # error = np.max(np.abs(a_grad_W-n_grad_W)) / np.maximum(1e-6,np.abs(np.max(a_grad_W)) + np.abs(np.max(n_grad_W)))
            # print(error)
    
            W -= eta*n_grad_W
            b -= eta*n_grad_b
        print('-- at end of epoch #{} --'.format(epoch))
        t_loss, t_cost, t_accuracy= computeCost(X,Y,W, b, lamda)
        v_loss, v_cost, v_accuracy= computeCost(X_v,Y_v, W, b, lamda)

        print(f'\t train loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')
        print(f'\t valid loss/cost/accuracy = {v_loss:.5g} / {v_cost:.5g} / {v_accuracy:.3g}')
        plot[epoch] = np.array([t_loss, t_cost, t_accuracy, v_loss, v_cost, v_accuracy])
	
    print('\n\nFinal results : ')
    t_loss,t_cost, t_accuracy = computeCost(test[0], test[1],W,b,lamda)
    print(f'\t test loss/cost/accuracy = {t_loss:.5g} / {t_cost:.5g} / {t_accuracy:.3g}')

    plt.subplot(121)
    plt.title('accuracy')
    plt.plot(range(1,n_epochs+1), plot[:,2], label="training")
    plt.plot(range(1,n_epochs+1), plot[:,5], label="validation")
    plt.legend()
   
    plt.subplot(122)
    plt.title('cost')
    plt.plot(range(1,n_epochs+1), plot[:,0], label="training")
    plt.plot(range(1,n_epochs+1), plot[:,3], label="validation")
    if(lamda>0):
	    plt.plot(range(1,n_epochs+1), plot[:,1], label="training")
	    plt.plot(range(1,n_epochs+1), plot[:,4], label="validation")
    plt.legend()
    title = f'lambda = {lamda}, n_epochs = {n_epochs}, n_batch = {n_batch}, eta = {eta}  '
    plt.suptitle(title)
    plt.show()

# resolveWithSDG(W, b)


# for report 
# resolveWithSDG(W, b, lamda=0, n_epochs=40, n_batch=100, eta=.1)
resolveWithSDG(W, b, lamda=0, n_epochs=40, n_batch=100, eta=.001)
# resolveWithSDG(W,b, lamda=.1, n_epochs=40, n_batch=100, eta=.001 )
# resolveWithSDG(W,b, lamda=1, n_epochs=40, n_batch=100, eta=.001 )
a = 2




