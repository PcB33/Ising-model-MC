# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:50:16 2020

@author: Phili
"""

import numpy as np
import matplotlib.pyplot as plt

N=np.linspace(1,100,100)
y=np.cos(np.pi/(2*N))**N

plt.plot(N,y)
plt.show()