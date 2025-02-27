# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:12:18 2025

@author: 20211382
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
red = 'tab:red'
blue= 'tab:blue'

def plot_equilib(file_path,num_steps,dist,temp,material,y1='Energy (eV)', y2=r'Cell Volume ($Å^3$)', save_fig=False):
    repeat=int(num_steps/dist+2)
    df_eng=pd.read_csv(file_path,header=None ,sep='\s+').iloc[1:repeat,:2]
    df_cell_v=pd.read_csv(file_path,header=None ,sep='\s+').iloc[repeat+1:,:2]

    eng_ar=df_eng.to_numpy(dtype=float)
    cel_ar=df_cell_v.to_numpy(dtype=float)
    
    
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel(y1, color=red)
    ax1.plot(eng_ar[:,0], eng_ar[:,1], color=red)
    ax1.tick_params(axis='y', labelcolor=red)
    
    ax2 = ax1.twinx()  
    
    ax2.set_ylabel(y2, color=blue)
    ax2.plot(cel_ar[:,0], cel_ar[:,1] ,color=blue)
    ax2.tick_params(axis='y', labelcolor=blue)
    
    fig.tight_layout()  
    plt.title(f'{material} - MD - {temp}K')
    plt.xlim(eng_ar[0,0], eng_ar[-1,0])
    if save_fig:
        plt.savefig(f'equilibrium_curve_{material}_{temp}K_{y1[0]}_{y2[0]}.jpg',dpi=300,bbox_inches='tight')
    plt.show()
    
def plot_MSD(file_path,res, temp, atoms_list,converged_D,c_list=[red,blue,'k'],save_fig=False):
    for i in range(len(atoms_list)):
        df_MSD=pd.read_csv(file_path,header=None ,sep='\s+').iloc[1+i*res:(i+1)*res,:2]
        x=df_MSD.iloc[:,0].to_numpy(dtype=float)
        MSD=df_MSD.iloc[:,1].to_numpy(dtype=float)*(0.529177)**2  
        plt.plot(x,MSD,label=f'Convergence {converged_D} m$^2$/s', c=c_list[i])
        
    plt.xlabel('Max Time (fs)')
    plt.ylabel(r'D (m$^2$/s)')
    plt.title(f'MSD - {atoms_list[0]} - MD - {temp}K')
    plt.legend()
    plt.xlim(x[0],x[-1])
    if save_fig:
        plt.savefig(f'MSD_{atoms_list[0]}_{temp}K.jpg',dpi=300,bbox_inches='tight')
    plt.show()

def plot_RDF(file_path,res, temp, atoms_list,save_fig=False,c_list=[red,blue,'k','g','orange','mediumorchid']):
    permutations=math.factorial(len(atoms_list))/math.factorial(len(atoms_list)-len(atoms_list))
    for i in range(int(permutations)):
        df_RDF=pd.read_csv(file_path,header=None ,sep='\s+').iloc[1+i*res:(i+1)*res,:2]
        r=df_RDF.iloc[:,0].to_numpy(dtype=float)
        g_r=df_RDF.iloc[:,1].to_numpy(dtype=float)
        if i<len(atoms_list):
            plt.plot(r,g_r,label=f'{atoms_list[0]} - {atoms_list[i]}', c=c_list[i], alpha=0.8)
        elif i<len(atoms_list)+(len(atoms_list)-1):
            index=int(len(atoms_list) - (permutations-len(atoms_list)+(len(atoms_list)-1)))
            plt.plot(r,g_r,label=f'{atoms_list[index]} - {atoms_list[int(-(permutations-i)+1)]}', c=c_list[i],alpha=0.8)
        else:
            plt.plot(r,g_r,label=f'{atoms_list[-1]} - {atoms_list[-1]}', c=c_list[i],alpha=0.8)
        
    plt.xlabel('r (Å)')
    plt.ylabel(r'g(r)')
    plt.title(f'RDF - {atoms_list[0]} - MD - {temp}K')
    #plt.legend()
    plt.xlim(r[0],r[-1])
    if save_fig:
        plt.savefig(f'RDF_{atoms_list[0]}_{temp}K.jpg',dpi=300,bbox_inches='tight')
    plt.show()
    return r,g_r

#90K 
#liquid   
num_steps=50000
dist=500
plot_equilib('Ar_equilibrium_curve.dat', num_steps, dist, '90', 'Liquid Argon',save_fig=True)
plot_MSD('Ar_liquid_MSD_90K.dat',51,'90',['Liquid Argon'],3.955e-9, save_fig=True)
r_l,g_r_l=plot_RDF('Ar_liquid_RDF_90K.dat', 1001, '90', ['Liquid Argon'])


#Crystal
n=100000
d=500
plot_equilib('Ar_crystal_90K_equilib.dat', n, d, '90', 'Crystal Argon',save_fig=True)
plot_MSD('Ar_crystal_90K_MSD.dat',51,'90',['Crystal Argon'],2.2642e-12, save_fig=True)
r_c,g_r_c=plot_RDF('Ar_crystal_RDF_90K.dat', 1001, '90', ['Crystal Argon'])

#Compare
plt.plot(r_l,g_r_l, c=red, label='Liquid')
plt.plot(r_c,g_r_c,c=blue,label='Crystal')
plt.xlabel('r (Å)')
plt.ylabel(r'g(r)')
plt.title('RDF - Compare Liquid to Crystal Argon - MD - 90K')
plt.legend()
plt.xlim(r_l[0],r_l[-1])
plt.savefig('RDF_Argon_L_v_C_90K.jpg',dpi=300,bbox_inches='tight')
plt.show()

#120K
#liquid
plot_equilib('Ar_liquid_120K_equilib.dat', 50000, 1000, '120', 'Liquid Argon',save_fig=True)
r_l120,g_r_l120=plot_RDF('Ar_liquid_120K_RDF.dat', 1001, '120', ['Liquid Argon'])
#Crystal
plot_equilib('Ar_crystal_120K_equilib.dat', n, d, '120', 'Crystal Argon',save_fig=True)
r_c120,g_r_c120=plot_RDF('Ar_crystal_120K_RDF.dat', 1001, '120', ['Crystal Argon'])

#Compare
plt.plot(r_l120,g_r_l120, c=red, label='Liquid')
plt.plot(r_c120,g_r_c120,c=blue,label='Crystal')
plt.xlabel('r (Å)')
plt.ylabel(r'g(r)')
plt.title('RDF - Compare Liquid to Crystal Argon - MD - 120K')
plt.legend()
plt.xlim(r_l120[0],r_l120[-1])
plt.savefig('RDF_Argon_L_v_C_120K.jpg',dpi=300,bbox_inches='tight')
plt.show()

#100K
plot_equilib('Ar_liquid_100K_equilib.dat', n, d, '100', 'Liquid Argon')
r_100,g_r_100=plot_RDF('Ar_liquid_100K_RDF.dat', 1001, '100', ['Liquid Argon'])

#110K
plot_equilib('Ar_liquid_110K_equilib.dat', n, d, '110', 'Liquid Argon')
r_110,g_r_110=plot_RDF('Ar_liquid_110K_RDF.dat', 1001, '110', ['Liquid Argon'])

#Compare Liquid Temperatures
plt.plot(r_l,g_r_l, c=red, label='90K', alpha=0.8)
plt.plot(r_100,g_r_100,c='k', label='100K', alpha=0.8)
plt.plot(r_110,g_r_110,c=blue, label='110K', alpha=0.8)
plt.plot(r_l120,g_r_l120, c='g', label='120K', alpha=0.8)
plt.xlabel('r (Å)')
plt.ylabel(r'g(r)')
plt.title('RDF - Varying Temperatures - MD - 120K')
plt.legend()
plt.xlim(r_l120[0],r_l120[-1])
plt.savefig('RDF_Argon_Liquid_vary_TEMP.jpg',dpi=300,bbox_inches='tight')
plt.show()

# LJ
eps_l=(119.8*8.3145)/1000
sigma_l=3.405
rs=np.linspace(2,12,100)
LJ_liquid=4*eps_l*((sigma_l/rs)**12-(sigma_l/rs)**6)

mol=1/(6.022e23)#1/(6.022e23)
well=6.626e-34*2.998e10*99.351
eps_dim=(well/mol)/1000
sigma_dim=3.762/(2**(1/6))
LJ_dimer=4*eps_dim*((sigma_dim/rs)**12-(sigma_dim/rs)**6)
plt.plot(rs,LJ_liquid,c=red, label='Liquid LJ')
plt.plot(rs,LJ_dimer,label='Dimer LJ')
plt.legend()
plt.ylabel('Energy (KJ/mol)')
plt.xlabel('r (A)')
plt.ylim(-2,1)
plt.savefig('LJ_curves_Dimer_Liquid.jpg',dpi=300,bbox_inches='tight')
plt.show()