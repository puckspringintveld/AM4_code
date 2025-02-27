# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:34:34 2025

@author: 20211382
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

colour = 'tab:red'

data=np.genfromtxt('G_OPT.dat')
m_pos=np.where(data[:,2]==data[0:,2].min())
cs=CubicSpline(data[:,0], data[:,2])
res=minimize(cs,data[:,0][m_pos])

plt.plot(data[:,0],data[:,2], '*', c=colour, label='Data')
plt.plot(np.linspace(data[:,0].min(), data[:,0].max(), 100), cs(np.linspace(data[:,0].min(), data[:,0].max(), 100)), c=colour, label='Interpolation')
plt.plot(res.x,cs(res.x), '*', c='k', label='Minimum')
plt.xlabel(r'Distance (Å)')
plt.ylabel(r'Energy (eV)')
plt.xlim(0.7,0.85)
plt.legend()
plt.savefig('Structure_opt_H2.jpg', dpi=300,bbox_inches='tight')
plt.show()

# Run 2
min_d=(0.1807047439929670-0.0292952560070338)*5


#Analysis
Exp=0.7414

nrmse_R1=np.sqrt((res.x[0]-Exp)**2/Exp**2)
nrmse_R2=np.sqrt((min_d-Exp)**2/Exp**2)

# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel(r'Distance (Å)')
# ax1.set_ylabel('Energy eV', color=color)
# ax1.plot(data[:,0],data[:,2],'-*' ,color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel(r'Force eV/ Å', color=color)  # we already handled the x-label with ax1
# ax2.plot(data[:,0],data[:,1],'--.' ,color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()