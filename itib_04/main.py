# Copyright 2021, Evula A. S., All rights reserved.
# IC8-63 BMSTU


from math import exp, sqrt
from matplotlib import pyplot as plt


class NN:
    def __init__(self):
        self.N = 1
        self.x = [1, 3]
        
        self.J = 2
        self.W_hidden = [[0] * (self.J) for _ in range(self.N+1)]
        
        self.M = 1        
        self.W_exit = [[0] * (self.M) for _ in range(self.J+1)]
        
        self.t = [0.1]
        self.Rate = 1
        
        self.MinError = 0.001
        self.MaxEpoch = 1000
        
    def train(self):
        epoch = 0
        E = 1
        self.res = [[], [], []]
        while E > self.MinError:
            if epoch > self.MaxEpoch:
                return print('ERR: epoch')
            res = self.tick()
            E, y = round(res[0], 4), res[1]
            self.res[0].append(epoch)
            self.res[1].append(round(E,4)*1000)
            self.res[2].append(y)
            epoch += 1
        return 0

    def tick(self):
        # Calculate y
        net_hidden = [0] * self.J
        
        for j in range(self.J):
            net_hidden[j] = self.W_hidden[0][j]
            for i in range(self.N):
                net_hidden[j] += self.W_hidden[i+1][j] * self.x[i+1]
       
        out = [] 
        for j in range(self.J):
            out.append(self.f_net(net_hidden[j]))
        
        net_exit = [0] * self.M
        for m in range(self.M):
            net_exit[m] = self.W_exit[0][m]
            for j in range(self.J):
                net_exit[m] += self.W_exit[j+1][m] * out[j]
        
        y = []
        for m in range(self.M):
            y.append(self.f_net(net_exit[m]))
        
        # Errors
        d_exit = []
        for m in range(self.M):
            d_exit.append(self.d_f_net(net_exit[m]) * (self.t[m]-y[m]))

        d_hidden = []
        for j in range(self.J):
            tmp = 0
            for m in range(self.M):
                tmp += d_exit[m] * self.W_exit[j][m]
            d_hidden.append(self.d_f_net(net_exit[m]) * tmp)
        
        # Correction
        for i in range(self.N+1):
            for j in range(self.J):
                self.W_hidden[i][j] += self.Rate*self.x[i]*d_hidden[j]
        
        out = [1] + out
        for j in range(self.J+1):
            for m in range(self.M):
                self.W_exit[j][m] += self.Rate*out[j]*d_exit[m]
            
        E = 0
        for m in range(self.M):
            E += (self.t[m]-y[m])*(self.t[m]-y[m])
        return sqrt(E), y
    
    def f_net(self, net):
        return (1-exp(-net))/(1+exp(-net))
    
    def d_f_net(self, net):
        return 0.5*(1 - self.f_net(net)*self.f_net(net))

    def print(self, graph=False, table=False):
        print(f't: {self.t}')
        
        print(end='\nВеса скрытого слоя\n')
        for j in range(len(self.W_hidden)):
            print(f'w{j}(1) = [', end='')
            for i in range(len(self.W_hidden[j])):
                print(round(self.W_hidden[j][i], 5), end='')
                if i != len(self.W_hidden[j])-1:
                    print(', ', end='')
            print(']')
        
        print(end='\nВеса выходного слоя\n')
        for j in range(len(self.W_exit)):
            print(f'w{j}(2) = [', end='')
            for i in range(len(self.W_exit[j])):
                print(round(self.W_exit[j][i], 5), end='')
                if i != len(self.W_exit[j])-1:
                    print(', ', end='')
            print(']')
        
        if graph:
            plt.rcParams["figure.figsize"] = (16,22)
            plt.plot(self.res[0], self.res[1], marker='o', markersize=4)
            plt.subplots_adjust(bottom = 0.15)
            plt.grid()
            plt.xlabel('epochs', fontsize=20)
            plt.ylabel('errors * 1000', fontsize=20)
            plt.xticks(self.res[0])
            plt.yticks(self.res[1])
            plt.margins(0.01)
            plt.show()
                    
        if table:
            print('Эпоха K\tВыходной Y')
            for i in range(len(self.res[0])):
                rounded_y = []
                for m in range(self.M):
                    rounded_y.append(round(self.res[2][i][m], 4))
                print(f'  {self.res[0][i]}\t  {rounded_y}')


def main():
    nn = NN()
    if nn.train() != None:
        nn.print(graph=True, table=True)


if __name__ == '__main__':
    main()

