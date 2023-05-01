from load import load 

class DatasetLoader:
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
	