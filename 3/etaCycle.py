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
