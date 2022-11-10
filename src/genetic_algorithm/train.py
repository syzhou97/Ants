# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/11/8 22:05 
@File : train.py
@Desc : 
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils import plot


class GeneticAlgorithm:
    def __init__(
        self,
        generations=100,
        population_size=100,
        mutation_rate=0.005,
        crossover_rate=0.8,
        scoring=None,
        x_range=[1, 5],
        x_num=1,
        dna_size=10,
        ax=None
    ):
        """

        :param generations: 迭代次数
        :param population_size: 种群大小
        :param mutation_rate: 变异概率
        :param crossover_rate: 交叉概率
        :param scoring: 适应度函数
        :param x_range: 自变量x的范围：[x_min, x_max]
        :param x_num: 自变量的个数
        :param dna_size: dna的大小
        :param ax:
        """
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scoring = scoring
        self.x_min, self.x_max = x_range
        self.x_num = x_num      # x的个数
        self.dna_size = dna_size
        self.ax = ax

        self.population = []
        self.fitness = np.zeros(self.dna_size)
        # 计算二进制位数
        self.best_x = []
        self.frames = []

    def run(self):
        self.init_population()
        for i in range(self.generations):
            x_decode = self.calc_fitness()
            if self.ax:
                if 'sca' in locals():
                    sca.remove()
                sca = self.ax.scatter(x_decode[0], x_decode[1], self.scoring(x_decode), c='black', marker='o')
                plt.show()
                plt.pause(0.1)
            self.select()
            self.cross()
            self.mutate()
        self.calc_fitness()
        max_fitness_index = np.argmax(self.fitness)
        print("max_fitness:", self.fitness[max_fitness_index])
        x_decode = self.decode_chromosome(self.population)
        print("最优的基因型：", self.population[max_fitness_index])
        for i in range(self.x_num):
            print("x"+str(i), x_decode[i][max_fitness_index])
            self.best_x.append(x_decode[i][max_fitness_index])

    def init_population(self):
        # 初始化
        self.population = np.random.randint(2, size=(self.population_size, self.dna_size*self.x_num))

    def decode_chromosome(self, x):
        # 解码
        x_decode = []
        for i in range(0, self.x_num):
            x_tmp = x[:, i*self.dna_size:(i+1)*self.dna_size]
            norm_x = x_tmp.dot(2**np.arange(self.dna_size)[::-1]) / float(2**self.dna_size-1)
            x_decode.append(norm_x * (self.x_max - self.x_min) + self.x_min)
        return x_decode

    def calc_fitness(self):
        x_decode = self.decode_chromosome(self.population)
        self.fitness = func(x_decode)
        self.fitness = self.fitness - np.min(self.fitness) + 1e-3           # 防止出现负数
        return x_decode

    def select(self):
        # 选择
        idx = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True, p=self.fitness/self.fitness.sum())
        self.population = self.population[idx]

    def cross(self):
        # 交叉：单点交叉
        # 两两配对
        pair = self.gen_pair()
        for i in range(self.population_size // 2):
            if random.random() < self.crossover_rate:
                ci1, ci2 = pair[i]
                # 随机生成交叉位
                bit_i = np.random.randint(0, self.dna_size * self.x_num)
                self.population[ci1][bit_i:], self.population[ci2][:bit_i] = self.population[ci2][bit_i:], self.population[ci2][:bit_i]

    def gen_pair(self):
        pair = []
        index = list(range(self.population_size))
        random.shuffle(index)
        for i in range(self.population_size):
            pair.append(index[i*2:(i+1)*2])
        return pair

    def mutate(self):
        # 变异
        for i in range(self.population_size):
            if random.random() < self.mutation_rate:
                mi = random.randint(0, self.dna_size * self.x_num-1)
                self.population[i][mi] = self.population[i][mi] ^ 1


def func1(x):
    # y = x1**2 + x2**2
    # s = 0
    # for i in x:
    #     s += i**2
    # return s
    x1, x2 = x[:2]
    return x1 ** 2 + x2 ** 2


def func2(x):
    # y=1/2*x+sin(3x)
    # return x[0]/2+math.sin(3*x[0])
    return x[0] / 2 + np.size(3 * x[0])


def func3(x):
    x, y = x[:2]
    return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) - 1/3**np.exp(-(x+1)**2 - y**2)


if __name__ == '__main__':
    func = func3
    x_min, x_max = -3, 3
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()
    plot.plot_3d(ax, x_min, x_max, func3)

    ga = GeneticAlgorithm(
        generations=200,
        population_size=200,
        mutation_rate=0.1,
        crossover_rate=0.8,
        scoring=func3,
        x_range=[x_min, x_max],
        x_num=2,
        dna_size=24,
        ax=ax
    )
    ga.run()
    print(ga.best_x)

    plt.ioff()
    plot.plot_3d(ax, x_min, x_max, func3)
    # gif.save(ga.frames, 'ga.gif', duration=3.5)