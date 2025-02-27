# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:16:56 2025

@author: 20211382
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
colour = 'tab:red'
THz=1e+12

# RUN0

run0=np.genfromtxt('Diamond/run0/G_OPT.dat')
m_pos=np.where(run0[:,2]==run0[0:,2].min())
cs=CubicSpline(run0[:,0], run0[:,2])
res=minimize(cs,run0[:,0][m_pos])

plt.plot(run0[:,0],run0[:,2],'*',c=colour, label='Data')
plt.plot(np.linspace(run0[:,0].min(), run0[:,0].max(), 100), cs(np.linspace(run0[:,0].min(), run0[:,0].max(), 100)), c=colour, label='Interpolation')
plt.plot(res.x,cs(res.x), '*', c='k', label='Minimum')
plt.xlabel("Lattice Constant")
plt.ylabel("Energy (eV)")
plt.legend()
plt.title("Lattice Constant Optimization Diamond")
plt.savefig('Lattice_Constant_Opt_Diamond.jpg', dpi=300,bbox_inches='tight')
plt.show()

# RUN 1
#take lattice constant from CONTCAR
min_l=3.80000000000000*0.9392506356981722

#Compare
Exp=3.57

nrmse_R1=np.sqrt((res.x[0]-Exp)**2/Exp**2)
nrmse_R2=np.sqrt((min_l-Exp)**2/Exp**2)

# RUN 2
DOS_c=np.genfromtxt('Diamond/run2/TDOS301.dat')
plt.plot(DOS_c[:,0], DOS_c[:,1], c=colour)
plt.xlabel("Energy (eV)")
plt.ylabel("DOS")
plt.title('Density of States for Diamond')
plt.xlim(DOS_c[:,0].min(), DOS_c[:,0].max())
plt.savefig('DOS_Diamond.jpg', dpi=300,bbox_inches='tight')
plt.show()

band_gap_c=4.11618 + 0.472823 #eV

# RUN 3
run3=np.genfromtxt('Diamond/run3/G_OPT.dat')

plt.plot(run3[:,0], run3[:,2], '-*', c=colour)
plt.xlabel('Ionic Step')
plt.ylabel('Energy (eV)')
plt.title('Diamond Energy Convergence')
plt.xlim(run3[:,0].min(), run3[:,0].max())
plt.savefig('Ionic_Opt_Diamond_after_distortion.jpg', dpi=300,bbox_inches='tight')
plt.show()

#compare equilibrium energies
difference=cs(res.x)-run3[-1,2]

# Si
# RUN 0
run0_si=np.genfromtxt('Si/run0/G_OPT.dat')
m_pos_si=np.where(run0_si[:,2]==run0_si[0:,2].min())
cs_si=CubicSpline(run0_si[:,0], run0_si[:,2])
res_si=minimize(cs_si,run0_si[:,0][m_pos_si])

plt.plot(run0_si[:,0],run0_si[:,2],'*',c=colour, label='Data')
plt.plot(np.linspace(run0_si[:,0].min(), run0_si[:,0].max(), 100), cs_si(np.linspace(run0_si[:,0].min(), run0_si[:,0].max(), 100)), c=colour, label='Interpolation')
plt.plot(res_si.x,cs_si(res_si.x), '*', c='k', label='Minimum')
plt.xlabel("Lattice Constant")
plt.ylabel("Energy (eV)")
plt.legend()
plt.title("Lattice Constant Optimization Si")
plt.savefig('Lattice_Constant_Opt_Si.jpg', dpi=300,bbox_inches='tight')
plt.show()

#RUN 1
min_l_si=5.70000000000000*0.9587115386995215

Exp_si=5.430941

nrmse_R1_si=np.sqrt((res_si.x[0]-Exp_si)**2/Exp_si**2)
nrmse_R2_si=np.sqrt((min_l_si-Exp_si)**2/Exp_si**2)

# RUN 2
DOS_si=np.genfromtxt('Si/run2/TDOS301.dat')
plt.plot(DOS_si[:,0], DOS_si[:,1], c=colour)
plt.xlabel("Energy (eV)")
plt.ylabel("DOS")
plt.title('Density of States for Sillicon')
plt.xlim(DOS_si[:,0].min(), DOS_si[:,0].max())
plt.savefig('DOS_Si.jpg', dpi=300,bbox_inches='tight')
plt.show()

band_gap_si=0.679794+0.126205

# RUN 3
run3_si=np.genfromtxt('Si/run3/G_OPT.dat')

plt.plot(run3_si[:,0], run3_si[:,2], '-*', c=colour)
plt.xlabel('Ionic Step')
plt.ylabel('Energy (eV)')
plt.title('Si Energy Convergence')
plt.xlim(run3_si[:,0].min(), run3_si[:,0].max())
plt.savefig('Ionic_Opt_Si_after_distortion.jpg', dpi=300,bbox_inches='tight')
plt.show()

#compare equilibrium energies
difference_si=cs_si(res_si.x)-run3_si[-1,2]

# AsAl
# RUN 1
min_l_AsAl=4.50000000000000*1.2733649224944719

# RUN 2
DOS_AsAl=np.genfromtxt('AsAl/run2/TDOS301.dat')
plt.plot(DOS_AsAl[:,0], DOS_AsAl[:,1], c=colour)
plt.xlabel("Energy (eV)")
plt.ylabel("DOS")
plt.title('Density of States for AsAl')
plt.show()

# RUN 4
import pandas as pd
df_real=pd.read_csv('AsAl/run4/real.dat',sep='\s+', header=None).iloc[:1000,:]
df_imag=pd.read_csv('AsAl/run4/imag.dat',sep='\s+', header=None).iloc[:1000,:]
df_real2=pd.read_csv('AsAl/run4/real.dat',sep='\s+', header=None).iloc[1001:,:]
df_imag2=pd.read_csv('AsAl/run4/imag.dat',sep='\s+', header=None).iloc[1001:,:]

n=np.array(np.sqrt((np.sqrt(df_real.iloc[:,1]**2+df_imag.iloc[:,1]**2)+df_real.iloc[:,1])/(2)))
n2=np.array(np.sqrt((np.sqrt(df_real2.iloc[:,1]**2+df_imag2.iloc[:,1]**2)+df_real2.iloc[:,1])/(2)))

wavelength=6.62607015*1e-34*299792458/(np.array(df_real.iloc[:,0],dtype=float)*1.60218e-19)*1e9
wavelength2=6.62607015*1e-34*299792458/(np.array(df_real2.iloc[:,0],dtype=float)*1.60218e-19)*1e9

plt.plot(wavelength,n, label='1', c='k')
plt.plot(wavelength2,n2, label='2', c=colour)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Refractive Index')
plt.xlim(450,950)
plt.legend()
plt.show()

