# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_toeplitz", "test_interp", "test_ski"]

import numpy as np
from scipy.linalg import circulant, toeplitz

from .solve import cg
from .ski import SKIMatrix
from .interp import LinearInterpolationMatrix
from .toeplitz import CirculantMatrix, ToeplitzMatrix


def test_toeplitz(N=50):
    print("Testing circulant linear algebra...")
    x = np.linspace(0, 10, N)
    y = np.vstack((np.sin(x), np.cos(x), x, x**2)).T
    c_row = np.exp(-0.5 * x ** 2)
    c_row[0] += 0.1
    cnum = circulant(c_row)
    cmat = CirculantMatrix(c_row)

    # Test dot products.
    assert np.allclose(np.dot(cnum, y[:, 0]), cmat.dot(y[:, 0]))
    assert np.allclose(np.dot(cnum, y), cmat.dot(y))

    # Test solves.
    assert np.allclose(np.linalg.solve(cnum, y[:, 0]), cmat.solve(y[:, 0]))
    assert np.allclose(np.linalg.solve(cnum, y), cmat.solve(y))

    print("Testing Toeplitz linear algebra...")
    tnum = toeplitz(c_row)
    tmat = ToeplitzMatrix(c_row)

    # Test dot products.
    assert np.allclose(np.dot(tnum, y[:, 0]), tmat.dot(y[:, 0]))
    assert np.allclose(np.dot(tnum, y), tmat.dot(y))

    # Test solves.
    assert np.allclose(np.linalg.solve(tnum, y[:, 0]),
                       tmat.solve(y[:, 0], tol=1e-12, verbose=True))
    assert np.allclose(np.linalg.solve(tnum, y),
                       tmat.solve(y, tol=1e-12, verbose=True))

    # Test logdet.
    _, logdet0 = np.linalg.slogdet(tnum)
    assert np.allclose(logdet0, tmat.logdet())


def test_interp(seed=1234, N=10, M=40):
    np.random.seed(seed)
    rng = (0, 2)
    x = np.sort(np.random.uniform(rng[0], rng[1], N))
    y = np.sin(x)
    x0 = np.linspace(rng[0]+0.05, rng[1], M)
    y0 = np.sin(x0)
    mat = LinearInterpolationMatrix(x0, x)
    assert np.allclose(mat.dot(y0), np.dot(mat.get_matrix(), y0))
    assert np.allclose(mat.dot(y, transpose=True),
                       np.dot(mat.get_matrix().T, y))


def test_ski(seed=1234, N=100, M=1000):
    def kernel(x, y):
        return np.exp(-0.5 * ((x - y) / 0.01) ** 2)

    np.random.seed(seed)
    rng = (0, 2)
    x = np.sort(np.random.uniform(rng[0], rng[1], N))
    y = np.sin(x)
    yerr = 0.05 * np.ones_like(y)
    u = np.linspace(rng[0], rng[1], M)

    mat = SKIMatrix(kernel, u, x, yerr)
    assert np.allclose(mat.W.dot(y, transpose=True),
                       mat.W.get_matrix().T.dot(y))
    assert np.allclose(mat.Kuu.dot(u), mat.Kuu.get_matrix().dot(u))
    assert np.allclose(mat.W.dot(u), mat.W.get_matrix().dot(u))
    assert np.allclose(mat.dot(y), mat.get_matrix().dot(y))

    r = cg(mat, y, np.zeros_like(y), verbose=True, tol=1e-8)
    r0 = np.linalg.solve(mat.get_matrix(), y)
    assert np.allclose(r, r0)
