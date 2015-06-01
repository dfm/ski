# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["SKIMatrix"]

import numpy as np

from .toeplitz import ToeplitzMatrix
from .interp import LinearInterpolationMatrix


class SKIMatrix(object):

    def __init__(self, kernel, u, x, yerr):
        self.W = LinearInterpolationMatrix(u, x)
        self.Kuu = ToeplitzMatrix(kernel(u[0], u))
        self.yvar = yerr ** 2

    def dot(self, y):
        K = self.W.dot(self.Kuu.dot(self.W.dot(y, transpose=True)))
        K += self.yvar * y
        return K

    def get_matrix(self):
        W = self.W.get_matrix()
        Kuu = self.Kuu.get_matrix()
        K = np.dot(W, np.dot(Kuu, W.T))
        K[np.diag_indices_from(K)] += self.yvar
        return K
