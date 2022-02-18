# Copyright 2021, Evula A. S., All rights reserved.
# IC8-63 BMSTU


from math import exp, sqrt
from matplotlib import pyplot as plt


class NN():
    def __init__(self):
        self.x = [ 
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ]
        self.t = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
        self.stable = [i for i in range(16)]

        zeros, ones = self.t.count(0), self.t.count(1)
        el = 0 if zeros<=ones else 1
        self.J = min(zeros, ones)

        self.c = [self.x[i] for i, e in enumerate(self.t) if e == el]

        self.W = [0] * (self.J+1)
        self.Rate = 0.3

        self.MaxEpoch = 20
        self.MinError  = 0


    def train(self):
        epoch = 0
        E = 1
        self.res = [[], []]
        while E > self.MinError:
            if epoch > self.MaxEpoch:
                return print('ERR: epoch')
            E = round(self.tick(), 3)
            self.res[0].append(epoch)
            self.res[1].append(E)
            epoch += 1
        return 0


    def tick(self):
        y = []
        
        y_K = []
        w_K = self.W.copy()

        for i in range(len(self.x)):
            phi = self.hiden(self.x[i])

            net = self.W[0]
            net_K = w_K[0]
            for j in range(len(phi)):
                net += self.W[j+1] * phi[j]
                net_K += w_K[j+1] * phi[j]

            y.append(int(net >= 0))
            y_K.append(int(net_K >= 0))

            self.correctWeights(phi=[1]+phi, delta=self.t[i]-y[i])

        E = 0
        for i in range(len(self.t)):
            E += 1 if self.t[i] != y_K[i] else 0 
        return E


    def hiden(self, x):
        res = []
        for j in range(self.J):
            s = 0
            for i in range(len(x)):
                s -= (x[i] - self.c[j][i]) * (x[i] - self.c[j][i])
            res.append(exp(s))
        return res


    def correctWeights(self, phi, delta):
        for j in range(len(self.W)):
            self.W[j] += self.Rate * delta * phi[j]


    def print(self, graph=False):
        for j in range(len(self.c)):
            print(f'c{j+1} = {self.c[j]}')

        print()
        for i in range(len(self.res[0])):
            print(f'K = {self.res[0][i]}\tE = {self.res[1][i]}')
        
        print()
        for j in range(len(self.W)):
            print(f'w{j} = {round(self.W[j], 2)}')
        
        if graph:
            plt.plot(self.res[0], self.res[1])
            plt.grid()
            plt.xlabel('epochs')
            plt.ylabel('errors')
            plt.xticks(self.res[0])
            plt.yticks(self.res[1])
            plt.savefig('graph.png')


def main():
    nn = NN()
    nn.train()
    nn.print(graph=True)


if __name__ == '__main__':
    main()