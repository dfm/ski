# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["SKIMatrix"]

import numpy as np

from .solve import cg, eig_pow_iter
from .toeplitz import ToeplitzMatrix
from .interp import LinearInterpolationMatrix


class SKIMatrix(object):

    def __init__(self, kernel, u, x, yerr):
        self.shape = (len(x), len(x))
        self.N = len(x)
        self.M = len(u)
        self.W = LinearInterpolationMatrix(u, x)
        self.Kuu = ToeplitzMatrix(kernel(u[0], u))
        self.yvar = float(yerr) ** 2

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

    def solve(self, other, **kwargs):
        other = np.atleast_1d(other)
        if other.shape[0] != self.N:
            raise ValueError("dimension mismatch")

        if len(other.shape) == 1:
            return cg(self, other, np.zeros_like(other), **kwargs)

        elif len(other.shape) == 2:
            r = np.empty_like(other)
            for i in range(other.shape[1]):
                r[:, i] = cg(self, other[:, i], np.zeros(other.shape[0]),
                             **kwargs)
            return r

        raise ValueError("unsupported dimensions")

    def logdet(self):
        f = self.N / self.M
        num = self.Kuu.get_matrix()
        print(np.abs(np.linalg.eigvals(num)[:15]))
        print(np.abs(self.Kuu.f_c[:15]))
        self.Kuu.logdet()
        assert 0

        # lam = f * self.Kuu.eigvals(sort=True, approx=True)[:self.N]
        # v = lam + self.yvar
        # if np.any(v <= 0.0):
        #     raise np.linalg.LinAlgError("the matrix is not positive definite")
        # return np.sum(np.log(v.real))
