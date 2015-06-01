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
yerr = 0.05 * np.ones_like(y)
u = np.linspace(rng[0], rng[1], 4000)

mat = SKIMatrix(kernel, u, x, yerr)

print(np.linalg.slogdet(mat.get_matrix()))
print(np.sum(np.log(100 / 4000. * mat.Kuu.f_c.real[:100] + yerr**2)))
