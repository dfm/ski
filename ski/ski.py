# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["SKIMatrix"]

import numpy as np
from numpy.linalg import LinAlgError
import scipy.sparse.linalg as splinalg
from scipy.sparse.linalg.interface import LinearOperator

from .solve import cg
from .toeplitz import ToeplitzMatrix
from .interp import LinearInterpolationMatrix


class SKIMatrix(LinearOperator):

    def __init__(self, kernel, u, x, yerr):
        self.shape = (len(x), len(x))
        self.N = len(x)
        self.M = len(u)
        self.W = LinearInterpolationMatrix(u, x)
        self.Kuu = ToeplitzMatrix(kernel(u[0], u))
        self.yvar = float(yerr) ** 2
        super(SKIMatrix, self).__init__(shape=self.shape, dtype=np.float64)

    def _matvec(self, y):
        K = self.W.dot(self.Kuu.dot(self.W.dot(y, transpose=True)))
        K += self.yvar * y
        return K

    def dot(self, y):
        return self._matvec(y)

    def get_matrix(self):
        W = self.W.get_matrix()
        Kuu = self.Kuu.get_matrix()
        K = np.dot(W, np.dot(Kuu, W.T))
        K[np.diag_indices_from(K)] += self.yvar
        return K

    def solve(self, other, **kwargs):
        result, flag = splinalg.cg(self, other, **kwargs)
        if flag:
            raise LinAlgError("conjugate gradient failed to converge")
        return result

    def eigvals(self, k=None, **kwargs):
        if k is None:
            k = self.N - 2
        return splinalg.eigs(self, return_eigenvectors=False, k=k, **kwargs)

    def logdet(self, **kwargs):
        return np.sum(np.log(np.abs(self.eigvals(**kwargs))))
