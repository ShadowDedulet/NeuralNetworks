# Copyright 2021, Evula A. S., All rights reserved.
# IC8-63 BMSTU


class NN:
	def __init__(self, rate):
		self.W = [0] * 3
		self.Rate = rate
		self.X = [
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]
		]

		self.t = [0, 1, 1, 0]

		self.border = 0


	def work(self):
		self.epoch = 0
		E = 1
		while (E > 0 and self.epoch < 101):
			print(f'''
---------------------------------
        Epoch: {self.epoch}
---------------------------------''')
			E = self.tick()
			self.epoch += 1

	def foo(self, net):
		return net

	def tick(self):
		y = [0] * len(self.X)
		y_k = [0] * len(self.X)
		w_k = self.W.copy()

		output = 'x  0 1 2   Y y t\n'
		for row in range(len(self.X)):
			x = self.X[row]
			
			net = self.W[0]
			net_k = w_k[0]
			for i in range(len(x)):
				net += x[i] * self.W[i+1]
				net_k += x[i] * w_k[i+1]

			out = int(self.foo(net) >= self.border)
			out_k = int(self.foo(net_k) >= self.border)

			y[row] = out
			y_k[row] = out_k

			output += '   1 '
			for i in x:
				output += str(i) + ' '
			output += '  ' + str(y[row]) + ' ' + str(y_k[row]) + ' ' +  str(self.t[row]) + '\t' + str(self.W)

			self.correctWeights(x, y[row], self.t[row])

			output += '\t' + str(self.W) + '\n'

		output += '\nW = '
		for w in self.W:
			output += str(round(w, 2)) + '  '

		E = self.dist(y_k)
		output += '\nE = ' + str(E)
		print(output)
		return E

	def correctWeights(self, x, y, t):
		x = [1] + x
		for i in range(len(self.W)):
			self.W[i] += self.Rate * (t - y) * x[i]

	def dist(self, y_K):
		e = 0
		for i in range(len(y_K)):
			e += abs(self.t[i] - y_K[i]) 
		return e

nn = NN(0.3)
nn.work()
# for i in range(1, 10):
# 	print(f'''
# =================================
#         RATE: {i/10}
# =================================''')
# 	nn = NN(i/10)
# 	nn.work()
