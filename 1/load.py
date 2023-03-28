import numpy as np

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
    return X, Y