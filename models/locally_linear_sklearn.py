#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:46:43 2023

@author: federico
"""

import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

loc_linear = LocallyLinearEmbedding(n_components=50)

X = np.random.randn(500,100)


loc_linear.fit_transform(X,)