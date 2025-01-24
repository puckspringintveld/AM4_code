import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

def Norm(psi,r):
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

#define dimension of M
M=11

#define range
m_range=range(1,M)

#populate A matrix
A=np.zeros([M-1,M-1])
for i, m in enumerate(m_range):
    A[i,:]= np.array([(math.factorial(m+x-2)*m*x
                         -math.factorial(m+x-1)*((x+m)/2)
                         +math.factorial(m+x)/4
                         -2*math.factorial(m+x-1))
                     for x in m_range])

#Populate S matrix
S=np.zeros([M-1,M-1])
for i, m in enumerate(m_range):
    S[i,:]=np.array([math.factorial(m+x)
                     for x in m_range])
 
#solve eigenvalue problem       
eig_vals,eig_vects=sp.linalg.eigh(A,S,subset_by_index=[0,2])

#Use coefficient matrix to transform
x_1=0
x_end=50
Nx=5000
eta=1e-3
x=eta+np.linspace(x_1,x_end,Nx)

chi=np.zeros([M-1,Nx])
for i, m in enumerate(m_range):
    chi[i,:]=x**m*np.exp(-1/2*x)
    

phi_1=eig_vects[:,0]@chi
phi_2=eig_vects[:,1]@chi
phi_3=eig_vects[:,2]@chi

#plot
plt.plot(x,Norm(phi_1,x)*phi_1,label=f'$\lambda_1$ = {np.round(eig_vals[0],3)}')
plt.plot(x,Norm(phi_2,x)*phi_2,label=f'$\lambda_2$ = {np.round(eig_vals[1],3)}')
plt.plot(x,Norm(phi_3,x)*phi_3,label=f'$\lambda_3$ = {np.round(eig_vals[2],3)}')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('Question_2b',dpi=300)
plt.show()