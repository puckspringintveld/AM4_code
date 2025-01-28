import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Define grid and potential
N = 100
a_s = np.linspace(-1, 1, N)
len_a = a_s[-1]
W0 = -1 / (2 * len_a**2)
ks=np.linspace(0,np.pi,10)
W0s=-np.linspace(np.abs(W0),1000,10)
test_conv=False
Vary_ks=True
Vary_pot=False

# Reciprocal space grid
Ns = np.linspace(-N//2, N//2 - 1, N)
G = Ns *2* np.pi / (len_a)
np.roll(G, shift=int(N/2))
A = np.zeros((N, N), dtype=complex)

#part a
for i, gn in enumerate(G):
    for j, gm in enumerate(G):
        if np.abs(gn - gm) < 1e-12:  # Diagonal
            A[i, j] = (gm)**2 - 2 * W0 / len_a
        else:  # Off-diagonal
            A[i, j] = -2 * W0 / len_a * (np.sin((gn - gm) * len_a / 4) / (gn - gm))

# Solve eigenvalue problem
eig_vals, eig_vects = np.linalg.eigh(A)
eigens=np.zeros([len(eig_vals),3], dtype=complex)


# Plot eigenvectors in reciprocal space
plt.figure(figsize=(8,6))
for i in range(3):
    eigens[:,i]=eig_vects[:,i]
    plt.plot(G, np.abs(eigens[:, i])**2, label=f"$\lambda_{i+1}$ = {eig_vals[i]:.3f}")
plt.title("Eigenfunctions in Reciprocal Space", fontsize=16)
plt.legend(fontsize=12)
plt.xlabel('Wave vector G',fontsize=16)
plt.ylabel('eigenfunction',fontsize=16)
plt.savefig('Question_4a_reciprocal', dpi=300)
plt.show()

# Map to real space using FFT
real_space_wavefunctions = np.fft.ifft(eig_vects,axis=0)
x = Ns*(len_a)/N

# Plot real-space wavefunctions
plt.figure(figsize=(9,6))
for i in range(3):
    plt.plot(x, np.abs(real_space_wavefunctions[:, i])**2, label=f"$\lambda_{i+1}$ = {eig_vals[i]:.3f}")
plt.title(f"Wavefunctions in Real Space",fontsize=16)
plt.legend(loc='upper left', bbox_to_anchor=(0.75, 1),fontsize=12)
plt.xlabel(f'$y$',fontsize=16)
plt.ylabel(f'$\chi^2(y)$',fontsize=14)
plt.xlim(-0.5,0.5)
plt.savefig('Question_4a_real', dpi=300)
plt.show()

plt.figure(figsize=(9,6))
for i in range(3):
    plt.plot(x, np.real(real_space_wavefunctions[:, i])**2, label=f"$\lambda_{i+1}$ = {eig_vals[i]:.3f}")
plt.title("Wavefunctions Real Component",fontsize=16)
plt.legend(loc='upper left', bbox_to_anchor=(0.75, 1),fontsize=12)
plt.xlabel(f'$y$',fontsize=16)
plt.ylabel(f'Re $\chi^2(y)$',fontsize=14)
plt.xlim(-0.5,0.5)
plt.savefig('Question_4a_real_p', dpi=300)
plt.show()
  
plt.figure(figsize=(9,6))  
for i in range(3):
    plt.plot(x, np.imag(real_space_wavefunctions[:, i])**2, label=f"$\lambda_{i+1}$ = {eig_vals[i]:.3f}")
plt.title("Wavefunctions Imaginary Component",fontsize=16)
plt.legend(loc='upper left', bbox_to_anchor=(0.75, 1),fontsize=12)
plt.xlabel(f'$y$',fontsize=16)
plt.ylabel(f'Im $\chi^2(y)$',fontsize=14)
plt.xlim(-0.5,0.5)
plt.savefig('Question_4a_im_p', dpi=300)
plt.show()

if test_conv==True:
    e_list=[]
    N_grid=np.linspace(10,1000,100)
    for ind, N in enumerate(N_grid):
        N=int(N)
        a_s = np.linspace(-1, 1, N)
        len_a = a_s[-1]
        W0 = 1 / (2 * len_a**2)
        ks=np.linspace(0,np.pi,10)
        W0s=np.linspace(W0,1000,10)
        
        # Reciprocal space grid
        Ns = np.linspace(-N//2, N//2 - 1, N)
        G = Ns *2* np.pi / (len_a)
        np.roll(G, shift=int(N/2))
        A = np.zeros((N, N), dtype=complex)

        #part a
        for i, gn in enumerate(G):
            for j, gm in enumerate(G):
                if np.abs(gn - gm) < 1e-12:  # Diagonal
                    A[i, j] = (gm)**2 - 2 * W0 / len_a
                else:  # Off-diagonal
                    A[i, j] = -2 * W0 / len_a * (np.sin((gn - gm) * len_a / 4) / (gn - gm))

        # Solve eigenvalue problem
        eig_vals, eig_vects = np.linalg.eigh(A)
        eigens=np.zeros([len(eig_vals),3], dtype=complex)
        
        e_list.append(eig_vals[0])
        
    plt.plot(N_grid, e_list-e_list[-1], '.')
    plt.xlabel('N')
    plt.ylabel('Convergence')
    plt.savefig('Question_4a_conv', dpi=300)
    plt.show()
        
        

if Vary_pot:
    #start W0 loop:
    lamdas=np.zeros((3,len(W0s)))
    Plot_imag_real=False
    Plot_coeffs_reciprocal=False
    Plot_coeffs_real=False
    for ind,W in enumerate(W0s):
        # Build Hamiltonian matrix
        for i, gn in enumerate(G):
            for j, gm in enumerate(G):
                if np.abs(gn - gm) < 1e-12:  # Diagonal
                    A[i, j] = (gm)**2 - 2 * W / len_a
                else:  # Off-diagonal
                    A[i, j] = -2 * W / len_a * (np.sin((gn - gm) * len_a / 4) / (gn - gm))
        
        # Solve eigenvalue problem
        eig_vals, eig_vects = np.linalg.eigh(A)
        eigens=np.zeros([len(eig_vals),3], dtype=complex)
        
        lamdas[:,ind]=eig_vals[:3]
    
        if Plot_coeffs_reciprocal:
            # Plot eigenvectors in reciprocal space
            for i in range(3):
                eigens[:,i]=eig_vects[:,i]
                plt.plot(G, np.abs(eigens[:, i])**2)
            plt.title("Eigenfunctions in Reciprocal Space")
            plt.show()
    
        # Map to real space using FFT
        real_space_wavefunctions = np.fft.ifft(eig_vects,axis=0)
        x = Ns*(len_a)/N
    
        if Plot_coeffs_real:
            # Plot real-space wavefunctions
            plt.figure(figsize=(9,6)) 
            for i in range(3):
                plt.plot(x, np.abs(real_space_wavefunctions[:, i])**2, label=f"$\lambda_{i}$ = {eig_vals[i]:.1f}")
            plt.title(f"At $V_0$={int(W)} : Wavefunctions in Real Space",fontsize=16)
            plt.legend(loc='upper left',bbox_to_anchor=(0.75, 1),fontsize=12)
            plt.xlabel(r'$y$',fontsize=16)
            plt.ylabel(r'$\chi^2(y)$',fontsize=14)
            plt.xlim(-0.5,0.5)
            plt.savefig('Question_4b_deep_well', dpi=300)
            plt.show()
        
        if Plot_imag_real:
            for i in range(3):
                plt.plot(x, np.real(real_space_wavefunctions[:, i])**2)
            plt.title("Wavefunctions Real Component")
            plt.show()
            
            for i in range(3):
                plt.plot(x, np.imag(real_space_wavefunctions[:, i])**2)
            plt.title("Wavefunctions Imaginary Component")
            plt.show()
    
    plt.plot(W0s,lamdas[0,:], '--*',label=r'$\lambda_1$')
    plt.plot(W0s,lamdas[1,:], '--*',label=r'$\lambda_2$')
    plt.plot(W0s,lamdas[2,:], '--*',label=r'$\lambda_3$')
    plt.xlabel(r'$V_0$')
    plt.ylabel('eigenvalue')
    plt.legend()
    plt.savefig('Question_4b_eig_V0', dpi=300)
    plt.show()

if Vary_ks:
    #start k loop:
    lamdas=np.zeros((3,len(ks)))
    Plot_imag_real=False
    Plot_coeffs_reciprocal=False
    Plot_coeffs_real=True
    for ind,k in enumerate(ks):
        # Build Hamiltonian matrix
        for i, gn in enumerate(G):
            for j, gm in enumerate(G):
                if np.abs(gn - gm) < 1e-12:  # Diagonal
                    A[i, j] = (k+gm)**2 - 2 * W0 / len_a
                else:  # Off-diagonal
                    A[i, j] = -2 * W0 / len_a * (np.sin((gn - gm) * len_a / 4) / (gn - gm))
        
        # Solve eigenvalue problem
        eig_vals, eig_vects = np.linalg.eigh(A)
        eigens=np.zeros([len(eig_vals),3], dtype=complex)
        
        lamdas[:,ind]=eig_vals[:3]
    
        if Plot_coeffs_reciprocal:
            # Plot eigenvectors in reciprocal space
            for i in range(3):
                eigens[:,i]=eig_vects[:,i]
                plt.plot(G, np.abs(eigens[:, i])**2)
            plt.title("Eigenfunctions in Reciprocal Space")
            plt.show()
    
        # Map to real space using FFT
        real_space_wavefunctions = np.fft.ifft(eig_vects,axis=0)
        x = Ns*(len_a)/N
    
        if Plot_coeffs_real:
            # Plot real-space wavefunctions
            plt.figure(figsize=(9,6)) 
            for i in range(3):
                plt.plot(x, np.abs(real_space_wavefunctions[:, i])**2, label=f"$\lambda_{i}$ = {eig_vals[i]:.1f}")
            #plt.title(f"k={np.round(k,3)} Wavefunctions in Real Space",fontsize=16)
            plt.title(f"At $k=\pi$ : Wavefunctions in Real Space",fontsize=16)
            plt.legend(loc='center left',bbox_to_anchor=(0.875, 0.5),fontsize=12)
            plt.xlabel(r'$y$',fontsize=16)
            plt.ylabel(r'$\chi^2(y)$',fontsize=14)
            plt.xlim(-0.5,0.5)
            plt.savefig('Question_4c_k_is_pi', dpi=300)
            plt.show()
        
        if Plot_imag_real:
            for i in range(3):
                plt.plot(x, np.real(real_space_wavefunctions[:, i])**2)
            plt.title("Wavefunctions Real Component")
            plt.show()
            
            for i in range(3):
                plt.plot(x, np.imag(real_space_wavefunctions[:, i])**2)
            plt.title("Wavefunctions Imaginary Component")
            plt.show()
    
    plt.plot(ks,lamdas[0,:],label=r'$\lambda_1$')
    plt.plot(ks,lamdas[1,:],label=r'$\lambda_2$')
    plt.plot(ks,lamdas[2,:],label=r'$\lambda_3$')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel('eigenvalue')
    plt.savefig('Question_4c_bandstructure', dpi=300)
    plt.show()