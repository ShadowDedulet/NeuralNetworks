# Copyright 2021, Evula A. S., All rights reserved.
# IC8-63 BMSTU


from math import sin, cos, exp, sqrt
from matplotlib import pyplot as plt

class NN:
    def __init__(self, p, rate, maxEpoch, minSigma):
        self.a = -5
        self.b = 3
        self.N = 20
        self.foo = lambda x: 0.5*exp(0.5*cos(0.5*x)) + sin(0.5*x)

        self.calc()

        self.W = [0] * (p+1)
        self.p = p
        self.rate = rate
        self.MAXEPOCH = maxEpoch
        self.MINSIGMA = minSigma

        self.train()

    def calc(self):
        self.c = 2*self.b - self.a
        self.d = (self.b-self.a)/(self.N-1)     # dist between points

        # correct foo
        X = [self.a + self.d*i for i in range(2*self.N - 1)]
        Y = [self.foo(x) for x in X]
        self.graph(X, Y)

    def train(self):
        self.delta = 1
        self.epoch = 0
        while self.delta > self.MINSIGMA and self.epoch < self.MAXEPOCH:
            self.tick()
            self.epoch += 1

    def tick(self):
        self.delta = 0
        for i in range(self.N - self.p + 1):
            y = [self.foo(self.a + (i+j)*self.d) for j in range(self.p)]
            
            net = self.W[0]
            for j in range(self.p):
                net += self.W[j+1] * y[j]

            sigma = self.foo(self.a + self.d*(i+self.p)) - net
            self.correctWeights(sigma, y)
            self.delta += sigma*sigma
        self.delta = sqrt(self.delta)
    
    def correctWeights(self, sigma, y):
        for i in range(self.p):
            self.W[i+1] += self.rate * sigma * y[i] 

    def predict(self):
        start = self.b - (self.d*self.p - 1)

        Y = []
        X = [start+self.d*(i+self.p) for i in range(self.N-1)]

        for i in range(self.N-1):
            y = [self.foo(start + (i+j)*self.d) for j in range(self.p)]

            net = self.W[0]
            for j in range(self.p):
                net += self.W[j+1] * y[j]
            Y.append(net)

        self.graph(X, Y)

    def graph(self, x, y):
        plt.plot(x, y)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')


def foo(_p, _rate):
    nn = NN(
        p = _p,
        rate = _rate,
        maxEpoch = 1000,
        minSigma = 0.05
    )
    nn.train()
    nn.predict()
    plt.show()
    print(f'window size: {round(nn.p, 2)}\nteaching rate: {round(nn.rate, 2)}')
    print(f'epochs: {round(nn.MAXEPOCH,2)}\nweights:', end=' ')
    for w in nn.W:
        print(round(w,2), end=' ')


if __name__ == '__main__':
    foo(4, 0.3)
    foo(4, 0.6)
    foo(6, 0.3)
    foo(6, 0.6)

window size: 4
teaching rate: 0.3
epochs: 1000
weights: 0 0.33 -0.84 -0.36 1.85 

window size: 6
teaching rate: 0.3
epochs: 1000
weights: 0 0.33 -0.25 -0.46 -0.28 0.32 1.32

window size: 6
teaching rate: 0.6
epochs: 1000
weights: 0 0.73 -0.27 -0.72 -0.55 0.23 1.62