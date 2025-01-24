# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:27:17 2024

@author: 20211382
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

x_1=0
x_end=50
Nx=5000
eta=1e-3
x=eta+np.linspace(x_1,x_end,Nx)
h=x_end/Nx
upper_diag=np.ones(Nx)*(-1/h**2)
diag=(2/h**2-2/x)*np.ones(Nx)
lower_diag=np.ones(Nx)*(-1/h**2)

def N(psi,r):
    """

    Parameters
    ----------
    psi : wave function to normalize
    r : space over normalization

    Returns
    -------
    Normalization constant

    """
    return 1/np.sqrt((np.trapz(psi**2,r)))


A=sp.sparse.spdiags([upper_diag, diag, lower_diag], [-1,0,1],Nx,Nx)
eig_vals,eig_vects=sp.sparse.linalg.eigs(A,sigma=-1.2)

plt.plot(x,N(eig_vects[:,0],x)*eig_vects[:,0],label=f'$\lambda_1$ = {np.round(eig_vals[0],3).real}')
plt.plot(x,N(eig_vects[:,1],x)*eig_vects[:,1],label=f'$\lambda_2$ = {np.round(eig_vals[1],3).real}')
plt.plot(x,N(eig_vects[:,2],x)*eig_vects[:,2],label=f'$\lambda_3$ = {np.round(eig_vals[2],3).real}')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('Question_1', dpi=300)
plt.show()


