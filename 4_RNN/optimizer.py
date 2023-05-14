import numpy as np
import matplotlib.pyplot as plt
from dataset import Data
from model import Model
from time import time
class Optimizer:
	def __init__(self, data, eta, m=100, verbose=True):
		self.data = data
		self.model = Model(data,eta, m)
		

	def compare_two_gradients_methods(self):
		from tqdm import tqdm
		seq_length = self.model.seq_length
		for seq_beg in tqdm(range(10)):
			input = self.data.seq_to_vec(self.data.book_data[seq_beg*seq_length:(seq_beg+1)*seq_length])
			output = self.data.seq_to_vec(self.data.book_data[seq_beg*seq_length+1:(seq_beg+1)*seq_length+1])
			
			a_s,h_s, pred = self.model.forward(input, h=np.zeros((self.model.m, 1)))
			grads_my_way = self.model.computeGrads(input, output, a_s, h_s, pred)
			grad_num = self.model.computeGradsNum(input, output)
			
			for layer_id, layer_name in enumerate(self.model.RNN):
				error = np.max(np.abs(grads_my_way[layer_id]-grad_num[layer_id])) / np.maximum(1e-6,np.abs(np.max(grads_my_way[layer_id])) + np.abs(np.max(grad_num[layer_id])))
				print(f'\terror is {error} for {layer_name}')
				

	def train(self, n_epochs):
		seq_length = self.model.seq_length
		synthesize_length = 200
		smooth_losses = []
		for epoch in range(n_epochs):
			print(f'\nepoch {epoch}\n')
			h_prev = np.zeros((self.model.m, 1))
			for seq_beg in range(1+len(self.data.book_data)//seq_length):
				input = self.data.seq_to_vec(self.data.book_data[seq_beg*seq_length:(seq_beg+1)*seq_length])
				output = self.data.seq_to_vec(self.data.book_data[seq_beg*seq_length+1:(seq_beg+1)*seq_length+1])

				a_s,h_s,pred = self.model.forward(input, h_prev)
				self.model.backpropagate(input, output, a_s, h_s, pred)

				seq_loss = self.model.computeLoss(output, pred)
				if(epoch == 0 and seq_beg == 0):
					smooth_losses.append(seq_loss)
				else:
					new_smooth_loss = .999*smooth_losses[-1] + .001*seq_loss
					smooth_losses.append(new_smooth_loss)
				
				if(seq_beg%200 == 0):
					iter = epoch*len(self.data.book_data)//seq_length + seq_beg
					print(f'iter = {iter} smooth_losses = {smooth_losses[-1]}')
				if(seq_beg%1000 == 0):
					input = self.data.seq_to_vec(self.data.book_data[seq_beg])
					synthesized_vec = self.model.synthesize(input, h_prev, synthesize_length)
					synthesized_seq = self.data.vec_to_seq(synthesized_vec)
					print(f"\n{''.join(synthesized_seq)} \n")
					

				h_prev = np.expand_dims(h_s[:,-1],1)
				#AdaGrad
				#SmoothLoss
				
		np.savetxt(f'losses_{time()}', smooth_losses)
			
			



# # evaluate test
# x0 = "a"
# x0 = data.seq_to_vec(x0)
# model.synthesize(x0, 25)


data = Data()


opt = Optimizer(data, 0.01,  m=5)
opt.compare_two_gradients_methods()

# opt = Optimizer(data, 0.1, m=100)
# opt.train(7)