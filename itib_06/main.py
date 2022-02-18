# Copyright 2021, Evula A. S., All rights reserved.
# IC8-63 BMSTU


import pandas as pd
from random import random
from math import sqrt, inf 


# liceum_key = {'name': 0, 'X': 1, 'Y': 2, 'district': 3}
# district_key = {'name': 0, 'X': 1, 'Y': 2}
class NN:
    def __init__(self, file_liceums, file_districts):
        self.liceums = pd.read_excel(file_liceums+'.xlsx', header=0, engine='openpyxl').values.tolist()
        self.districts = pd.read_excel(file_districts+'.xlsx', header=0, engine='openpyxl').drop(columns=['Unnamed: 0']).values.tolist()

        self.Rate = 0.3
        self.W = []
        for _ in range(len(self.districts)):
            temp = []
            for _ in range(len(self.liceums)):
                            temp.append([round(random(), 2), round(random(), 2)])
            self.W.append(temp) 

    def train(self, debug=False):
        epoch = -1
        error = inf
        while error > 0 and epoch < 1000:
            epoch += 1
            error = self.tick(debug)
            print(f'\nK: {epoch}')
            print(f'E: {error}/{len(self.liceums)}')

    def tick(self, debug):
        error = 0
        for i in range(len(self.liceums)):
            t = [0] * len(self.districts)
            t[self.find_district(self.liceums[i])] = 1
            y = [0] * len(self.districts)

            min_dist = inf
            min_j = 0
            for j in range(len(self.districts)):
                Pj = self.get_dist(self.liceums[i], self.districts[j], i, j)
                if min_dist > Pj:
                    min_dist = Pj
                    min_j = j
            y[min_j] = 1

            if debug:
                print(self.liceums[i][0])
                print(f'\tCorrect: {self.liceums[i][3]}')
                print(f'\tPredict: {self.districts[min_j][0]}')

            if y != t:
                error += 1
                self.correct(i, t, y)
        return error   

    def get_dist(self, liceum, district, i, j):
        l_coords = (liceum[1], liceum[2])
        d_coords = (district[1], district[2])
        temp = self.W[j][i][0]*l_coords[0] + self.W[j][i][1]*l_coords[1] 

        return 0.5*self.vector_length(d_coords) - temp

    def vector_length(self, vec):
        return vec[0]*vec[0] + vec[1]*vec[1]

    def correct(self, i, t, y):
        for j in range(len(t)):
            if t[j] == y[j]:
                continue
            self.W[j][i][0] += (t[j]-y[j])*self.Rate*self.liceums[i][1]
            self.W[j][i][1] += (t[j]-y[j])*self.Rate*self.liceums[i][2]

    def find_district(self, liceum):
        district = None
        for j in range(len(self.districts)):
            if liceum[3] == self.districts[j][0]:
                return j


def main():
    nn = NN(file_liceums='./liceums', file_districts='./districts')
    nn.train(debug=False)


if __name__ == '__main__':
    main()