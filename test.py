#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
np.random.seed(123)

from ski.ski import SKIMatrix
from ski.solve import cg


def kernel(x, y):
    return np.exp(-0.5 * ((x - y) / 0.01) ** 2)


rng = (0, 2)
x = np.sort(np.random.uniform(rng[0], rng[1], 100))
y = np.sin(x)
yerr = 0.05
u = np.linspace(rng[0], rng[1], 1000)

mat = SKIMatrix(kernel, u, x, yerr)
num = mat.get_matrix()

n = 15
# print(mat.Kuu.eigvals()[:n])
# print(mat.Kuu.eigvals(approx=False)[:n])

print(np.linalg.slogdet(num))
print(np.sum(np.log(mat.eigvals())))
