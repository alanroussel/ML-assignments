import numpy as np
from dataset import Data
from model import Model
from time import time
class Optimizer:
	def __init__(self, data, debug_gradients=False):
		self.data = data
		if(debug_gradients):
			self.model = Model(data.K, m=5)
		else:
			self.model =  Model(data.K, m=100)
		

	def compare_two_gradients_methods(self):
		seq_length = self.model.seq_length
		for seq_beg in range(10):
			print()
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
				
				if(seq_beg%10000 == 0):
					iter = epoch*len(self.data.book_data)//seq_length + seq_beg
					print(f'iter = {iter} smooth_losses = {smooth_losses[-1]}')
				if(seq_beg%10000 == 0):
					input = self.data.seq_to_vec(self.data.book_data[seq_beg])
					synthesized_vec = self.model.synthesize(input, h_prev, synthesize_length)
					synthesized_seq = self.data.vec_to_seq(synthesized_vec)
					print(f"\n{''.join(synthesized_seq)} \n")
					

				h_prev = np.expand_dims(h_s[:,-1],1)
				#AdaGrad
				#SmoothLoss
		self.model.save(n_epochs)
		np.savetxt(f'losses_{time()}', smooth_losses)
			
			



# # evaluate test
# x0 = "a"
# x0 = data.seq_to_vec(x0)
# model.synthesize(x0, 25)


data = Data()

mode = "testing" # "training", or "testing"
n_epochs_to_learn = 3

if(mode=="debug"):
	opt = Optimizer(data, debug_gradients=True)
	opt.compare_two_gradients_methods()
elif(mode=="training"):
	opt = Optimizer(data)
	opt.train(n_epochs_to_learn)
elif(mode=="testing"):
	model = Model(data.K, m=100)
	model.load(n_epochs_to_learn)

	
	input = data.seq_to_vec(data.book_data[0])
	output = data.seq_to_vec(data.book_data[0:1000])
	print(''.join(data.book_data[0:1000]))
	for i in range(10):
		h = np.zeros((model.m, 1))
		
		synthesized_vec = model.synthesize(input, h, n=1000)
		synthesized_seq = data.vec_to_seq(synthesized_vec)
		print(f"\n{''.join(synthesized_seq)} \n")
		


		
