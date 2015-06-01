# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["CirculantMatrix", "ToeplitzMatrix"]

import numpy as np

from .solve import cg

# Python 3.
try:
    xrange
except NameError:
    xrange = range


class CirculantMatrix(object):

    def __init__(self, row):
        self.row = row
        self.f_row = np.fft.rfft(row)

    def __mul__(self, other):
        return self.dot(other)

    def dot(self, other):
        other = np.atleast_1d(other)
        if other.shape[0] != len(self.row):
            raise ValueError("dimension mismatch")
        if len(other.shape) == 1:
            return np.fft.irfft(self.f_row * np.fft.rfft(other))
        elif len(other.shape) == 2:
            return np.fft.irfft(self.f_row[:, None]*np.fft.rfft(other, axis=0),
                                axis=0)
        raise ValueError("unsupported dimensions")

    def solve(self, other):
        other = np.atleast_1d(other)
        if other.shape[0] != len(self.row):
            raise ValueError("dimension mismatch")
        if len(other.shape) == 1:
            return np.fft.irfft(np.fft.rfft(other) / self.f_row)
        elif len(other.shape) == 2:
            return np.fft.irfft(np.fft.rfft(other, axis=0)/self.f_row[:, None],
                                axis=0)
        raise ValueError("unsupported dimensions")


class ToeplitzMatrix(object):

    def __init__(self, row):
        self.row = row
        c = np.append(row, row[1:-1][::-1])
        self.f_c = np.fft.rfft(c)
        self.N = len(row)
        self.dim = 2 * (self.N - 1)

    def get_matrix(self):
        from scipy.linalg import toeplitz
        return toeplitz(self.row)

    def __mul__(self, other):
        return self.dot(other)

    def dot(self, other):
        other = np.atleast_1d(other)
        if other.shape[0] != self.N:
            raise ValueError("dimension mismatch")

        if len(other.shape) == 1:
            z = np.zeros(self.dim)
            z[:self.N] = other
            return np.fft.irfft(self.f_c * np.fft.rfft(z))[:self.N]

        elif len(other.shape) == 2:
            z = np.zeros((self.dim, other.shape[1]))
            z[:self.N] = other
            return np.fft.irfft(self.f_c[:, None]*np.fft.rfft(z, axis=0),
                                axis=0)[:self.N]

        raise ValueError("unsupported dimensions")

    def solve(self, other, **kwargs):
        other = np.atleast_1d(other)
        if other.shape[0] != self.N:
            raise ValueError("dimension mismatch")

        if len(other.shape) == 1:
            return cg(self, other, np.zeros_like(other), **kwargs)

        elif len(other.shape) == 2:
            r = np.empty_like(other)
            for i in xrange(other.shape[1]):
                r[:, i] = cg(self, other[:, i], np.zeros(other.shape[0]),
                             **kwargs)
            return r

        raise ValueError("unsupported dimensions")

    def logdet(self):
        n = self.N - 1
        lam = np.zeros(n)
        ghat = np.zeros(n)
        gamma = np.zeros(n)

        r = self.row[1:] / self.row[0]
        lam[0] = 1 - r[0] ** 2
        ghat[0] = -r[0]

        for i in xrange(n-1):
            gamma[i] = -r[i+1]
            gamma[i] -= np.sum(r[:i+1] * ghat[:i+1])
            ghat[1:i+2] = ghat[:i+1] + gamma[i] / lam[i] * ghat[:i+1][::-1]
            ghat[0] = gamma[i] / lam[i]
            lam[i+1] = lam[i] - gamma[i]**2 / lam[i]

        return np.sum(np.log(lam)) + (n+1) * np.log(self.row[0])
