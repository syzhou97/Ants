# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/11/9 11:39 
@File : plot.py 
@Desc : 
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(ax, x_min, x_max, func):
    X = np.linspace(x_min, x_max, 100)
    Y = np.linspace(x_min, x_max, 100)
    X, Y = np.meshgrid(X, Y)
    Z = func([X, Y])
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm)
    ax.set_zlim(-10,10)
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('y')
    plt.pause(3)
    plt.show()
