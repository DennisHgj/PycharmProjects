import numpy as np
from math import pi,exp

#记录√（2π），避免该项的重复运算
sqrt_pi = (2 * pi) ** 0.5

class NBFunctions:
    """定义正态分布的函数"""
    @staticmethod
    def gaussian(x, mu, sigma):
        return exp(-(x - mu) ** 2 / (2 * sigma ** 2))/(sqrt_pi * sigma)

    """定义极大似然估计的函数
    它能返回一共存储着计算条件概率密度的函数的列表"""

    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):
        mu = [np.sum(
            labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum(
            (labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]

        def func(_c):
            def sub(x):
                return NBFunctions.gaussian(x, mu[_c], sigma[_c])

            return sub

        return [func(_c=c) for c in range(n_category)]