# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["cg", "eig_pow_iter"]

import numpy as np

# Python 3.
try:
    xrange
except NameError:
    xrange = range


def cg(A, b, x0=None, M=None, maxiter=400, verbose=False, tol=1e-6):
    """
    Algorithm from pages 529 and 534 of "Matrix Computations" by Golub

    """
    if x0 is None:
        x0 = np.zeros_like(b)
    if M is None:
        M = lambda _: np.array(_)

    x = np.array(x0)
    r = b - A.dot(x)
    oldrho = 0.0
    for i in xrange(maxiter):
        z = M(r)
        rho = np.dot(r.T, z)
        if np.sqrt(rho) < tol:
            if verbose:
                print(i)
            return x
        if i == 0:
            p = z
        else:
            beta = rho / oldrho
            p = z + beta * p
        q = A.dot(p)
        alpha = rho / np.dot(p.T, q)
        x = x + alpha * p
        r = r - alpha * q
        oldrho = rho

    raise np.linalg.LinAlgError("conjugate gradient solve failed to converge")
