# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:33:25 2025

@author: 20211382
"""

import numpy as np
import matplotlib.pyplot as plt
colour = 'tab:red'
THz=1e12

# H2O
energy_H2O=np.genfromtxt('H2O/G_OPT.dat')
vibrations_H2O=np.genfromtxt('H2O/vibrations.dat') #tera hertz

plt.plot(energy_H2O[:,0], energy_H2O[:,2], '-*' ,c=colour, label=f'Final Energy: {np.round(energy_H2O[-1,2],3)}')
plt.xlabel('Ionic Step')
plt.ylabel('Energy (eV)')
plt.legend()
plt.title(r'Geometry Optimization of H$_2$O')
plt.savefig('Ionic_opt_H2O.jpg', dpi=300,bbox_inches='tight')
plt.show()

print(f'Vibrations of water: \n HO assymetric stretching:{vibrations_H2O[0,2]*THz/1e14} x 10^14 \n HO symetric stretching:{vibrations_H2O[1,2]*THz/1e14} x 10^14 \n bending:{vibrations_H2O[2,2]*THz/1e13} x 10^13' )

# CO2
energy_CO2=np.genfromtxt('CO2/run1/G_OPT.dat')
vibrations_CO2=np.genfromtxt('CO2/run2/vibrations.dat') #tera hertz

plt.plot(energy_CO2[:,0], energy_CO2[:,2], '-*' ,c=colour, label=f'Final Energy: {np.round(energy_CO2[-1,2],3)}')
plt.xlabel('Ionic Step')
plt.ylabel('Energy (eV)')
plt.legend()
plt.title(r'Geometry Optimization of CO$_2$')
plt.savefig('Ionic_opt_CO2.jpg', dpi=300,bbox_inches='tight')
plt.show()

print(f'Vibrations of carbon dioxide: \n CO assymetric stretching:{vibrations_CO2[0,2]*THz/1e14} x 10^14 \n CO symetric stretching:{vibrations_CO2[1,2]*THz/1e14} x 10^14 \n bending:{vibrations_CO2[2,2]*THz/1e13} x 10^13' )