# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:19:19 2025

@author: 20211382
"""

import numpy as np
import matplotlib.pyplot as plt
THz=1e+12

freq=117.663050*THz
print(f'vibrations in H2: {freq/1e14}e14 Hz')

data=np.genfromtxt('TDOS301.dat')
colour = 'tab:red'

plt.plot(data[:,0], data[:,1], c=colour)
plt.xlabel("Energy (eV)")
plt.ylabel("DOS")
plt.xlim(data[:,0].min(), data[:,0].max())
plt.savefig('DOS_H2.jpg', dpi=300,bbox_inches='tight')
plt.show()