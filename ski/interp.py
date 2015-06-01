# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["LinearInterpolationMatrix"]

import numpy as np


class LinearInterpolationMatrix(object):

    def __init__(self, x0, x):
        self.shape = (len(x), len(x0))

        # Compute the neighbor indices.
        hi = np.searchsorted(x0, x)
        lo = hi - 1
        lo = np.clip(lo, 0, len(x0) - 1)
        hi = np.clip(hi, 0, len(x0) - 1)

        # Compute the weights.
        m = hi > lo
        w = np.zeros(len(m))
        w[m] = (x[m] - x0[lo][m]) / (x0[hi][m] - x0[lo][m])

        # Save the values.
        self.hi = hi
        self.lo = lo
        self.w = w
        self.omw = 1. - w

    def get_matrix(self):
        W = np.zeros(self.shape)
        W[np.arange(self.shape[0]), self.hi] += self.w
        W[np.arange(self.shape[0]), self.lo] += self.omw
        return W

    def dot(self, y, transpose=False):
        y = np.atleast_1d(y)
        if transpose:
            if len(y) != self.shape[0]:
                raise ValueError("incompatible dimensions")
            r = np.zeros(self.shape[1])
            np.add.at(r, self.lo, self.omw * y)
            np.add.at(r, self.hi, self.w * y)
            return r

        if len(y) != self.shape[1]:
            raise ValueError("incompatible dimensions")
        return self.w * y[self.hi] + self.omw * y[self.lo]
