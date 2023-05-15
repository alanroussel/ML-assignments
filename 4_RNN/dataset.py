import numpy as np

class Data:
	def __init__(self):
		with open('goblet_book.txt', 'r') as file:
			content = file.read()
			self.book_data = np.array(list(content))
			self.book_chars = np.unique(self.book_data)
			self.K = len(self.book_chars)

			self.dict_int_to_char = dict(zip( np.arange(self.K), self.book_chars))
			self.dict_char_to_int = dict(zip(self.book_chars,  np.arange(self.K)))


	def seq_to_vec(self, y):
		y_encoded = [self.dict_char_to_int[char] for char in y]
		vec = np.zeros((len(y_encoded), self.K))
		vec[np.arange(len(y_encoded)), y_encoded] = 1
		return vec.T

	def vec_to_seq(self, vec):
		ints = np.argmax(vec.T, 1)
		seq = [self.dict_int_to_char[int] for int in ints]
		return seq

data = Data()
y = ["a", "l", "a", "n"]
a = 2