#!/usr/bin/python3

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

def ElectronDensity(wavefunc):
	"""
	Function:	function that calculates the electron density from the orbital wave function.
	Input:		wavefunc: a 1-dimensional vector that stores the wavefunction.
	Output:		dens: a 1-dimensional vector that stores the electron density.
	Errors:		none, you're on your own.
	! NOTE: A real space projection is done since the wavefunctions are purely real and this gets rid of warnings !
	"""
	dens = np.real(np.square(wavefunc))

	return dens

def HartreePotential(y_n, step, dens):
	"""
	Function:	calculates the Hatree potential from a given electron density on a numerical grid
	Input:		y_n: the numerical grid used in the calculation.
				step: the step size of the grid.
				dens: the electron density of the system.
	Output:		hartree: the Hartree potential for the given electron density of the system.
	Errors:		none, you're on your own.
	"""
	N_grid = len(y_n)
	charge = np.cumsum(dens) * step
	hartree = np.zeros(N_grid)

	pot_int = 0

	for i in range(N_grid - 2, -1, -1):
		pot_int = pot_int + 2 * charge[i + 1] / (y_n[i + 1] ** 2) * step
		hartree[i] = hartree[i] + pot_int

	hartree = hartree + 2 / y_n[-1]

	return hartree

def Norm(array,x):
    """

    Parameters
    ----------
    array : density to normalize
    x : space over normalization

    Returns
    -------
    Normalization constant

    """
    integral=np.trapz(array,x)
    return 1/integral

# grid parameters (see problem 1)
eta = 1E-10
N_grid = 10000
y_end = 10.0
y = np.linspace(0, y_end, N_grid) + eta
h = y[1] - y[0]

# construct matrix elements, kinetic energy and nuclear potential (see problem 1)
subd = -1 * np.ones(N_grid) / h ** 2
superd = -1 * np.ones(N_grid) / h ** 2
diagonal = 2 * ((1 / h ** 2) - 2 / y)

# initialize wave function, density and Hartree potential
chi = np.exp(- y / 2) * y / np.sqrt(2)
dens = ElectronDensity(chi)
hartree = HartreePotential(y, h, dens)

print(np.trapz(chi*chi,y))

############################################################
### Insert your own code here for problems 6(b) and 6(c) ###
SCF_steps=50
iteration=0
conv=1
tol=1e-6
eigens=[0]
while iteration <= SCF_steps and np.abs(conv)>tol:
    A=spdiags([subd,diagonal+hartree,superd], [-1,0,1]) #construct matrix
    eig_vals,chi_new=eigs(A,k=1, sigma=-1.8) #solve eigenvalue problem
    conv=np.real(eig_vals[0])-np.real(eigens[-1]) #test convergence 
    eigens.append(eig_vals[0]) #save eigenvalue for convergence analysis
    dens=ElectronDensity(chi_new) #generate new density
    N=Norm(dens[:,0], y) #find normalization constant
    dens=dens*N #normalize new density
    chi_new=chi_new*np.sqrt(N) #normalize new wave function
    hartree_new = HartreePotential(y, h, dens) #calculate new Hartree pot
    hartree=hartree_new
    iteration+=1 #count loop iteration

plt.plot(np.zeros(14),'--' ,c='k', alpha=0.5)
plt.plot(np.array(eigens[1:]).real+1.836, '-*')
plt.ylabel('Convergence')
plt.xlabel('Iteration Cycle')
plt.xlim(-0.1,13.1)
plt.savefig('Question_6a_conv', dpi=300)
plt.show()

print(np.trapz(chi_new[:,0]*chi_new[:,0],y))

plt.plot(y,chi_new[:,0], label=f'$\lambda$ ={np.round(eig_vals[0].real,3)}')
plt.xlabel('y')
plt.ylabel(r'$\chi(y)$')
plt.xlim(-0.1,10)
plt.legend()
plt.savefig('Question_6a', dpi=300)
plt.show()

    
############################################################

############################################################
### Insert your own code here for problems 6(b) and 6(c) ###
############################################################
rho=chi_new[:,0]**2
E_1=2*eig_vals[0]-np.trapz(hartree*rho,y)#-0.5*np.trapz(rho*rho/(4*np.pi*eig_vals[0]*y),y)

eig,chis=eigs(A,sigma=-1.8)

E1_ionization=eig[0]-eig[1]
E2_ionization=eig[0]-eig[2]
