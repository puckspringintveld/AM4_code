# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:02:00 2025

@author: 20211382
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from scipy.signal import savgol_filter
red = 'tab:red'
blue= 'tab:blue'

num_steps=50000
dist=500

def R2(Y_real,Y_fit):
    Y_mean=Y_real.mean()
    R2=1-np.sum((Y_real-Y_fit)**2)/np.sum((Y_real-Y_mean)**2)
    return R2

def smoothing(X,Y,type_filter='Linear_Regression', testing=False): 
    if type_filter=='LR':
        res = stats.linregress(X, Y)
        new_Y=res.intercept + res.slope*X
        return new_Y, res.rvalue**2
    elif type_filter=='US':
        us = UnivariateSpline(X, Y)
        return us(X),R2(Y,us(X))
    elif type_filter=='GF':
        new_Y=gaussian_filter(Y, sigma=5)
        return new_Y, R2(Y,new_Y)
    elif type_filter=='GPR':
        kernel= ConstantKernel() + Matern((X.reshape(len(X),1).max()*0.2),[(X.reshape(len(X),1).min()+0.05,100*X.reshape(len(X),1).max())],2.5) + WhiteKernel(noise_level=0.1)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
        print(gp.fit(X.reshape(len(X),1), Y.reshape(len(X),1)))
        print(gp.kernel_)
        new_Y = gp.predict(X.reshape(len(X),1), return_std=False)
        return new_Y, R2(Y,new_Y)
    elif type_filter=='SG':
        new_Y=savgol_filter(Y, 40, 2)
        return new_Y, R2(Y,new_Y)
                    
def tesing_filters(file_path,num_steps,dist, save_fig=False):
    repeat=int(num_steps/dist+2)
    df_LV1=pd.read_csv(file_path,header=None ,sep='\s+').iloc[1:repeat,:2]
    df_LV2=pd.read_csv(file_path,header=None ,sep='\s+').iloc[repeat+1:2*repeat,:2]
    df_LV3=pd.read_csv(file_path,header=None ,sep='\s+').iloc[2*repeat+1:3*repeat,:2]
    
    LV1=df_LV1.to_numpy(dtype=float)
    LV2=df_LV2.to_numpy(dtype=float)
    LV3=df_LV3.to_numpy(dtype=float)
    
    plt.plot(LV1[:,0],np.sqrt(LV1[:,1]**2+LV2[:,1]**2+LV3[:,1]**2),'.' ,c='k', label='Noisy Data')
    list_types=['LR','US', 'GF',  'GPR', 'SG']
    for i, type_f in enumerate(list_types):
        smooth_LV,R2=smoothing(LV1[:,0],np.sqrt(LV1[:,1]**2+LV2[:,1]**2+LV3[:,1]**2),type_filter=type_f)    
        plt.plot(LV1[:,0],smooth_LV, label=f'{list_types[i]}, R$^2$={np.round(R2,3)}')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Lattice Vector (Å)')
    plt.xlim(LV1[0,0], LV1[-1,0])
    plt.legend()
    plt.title('Testing Filters')
    if save_fig:
        plt.savefig('Testing_Filters_on_Lattice_Vector.jpg',dpi=300,bbox_inches='tight')
    plt.show()           

def plot_equilib(file_path,num_steps,dist,temp,material,y1='Energy (eV)', y2=r'Cell Volume ($Å^3$)',smooth=False, save_fig=False):
    repeat=int(num_steps/dist+2)
    df_eng=pd.read_csv(file_path,header=None ,sep='\s+').iloc[1:repeat,:2]
    df_cell_v=pd.read_csv(file_path,header=None ,sep='\s+').iloc[repeat+1:,:2]

    eng_ar=df_eng.to_numpy(dtype=float)
    cel_ar=df_cell_v.to_numpy(dtype=float)
    
    if smooth:
        smooth_cel_ar,_=smoothing(cel_ar[:,0]/1000, cel_ar[:,1]/1000, type_filter='GPR')
    
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel(y1, color=red)
    ax1.plot(eng_ar[:,0], eng_ar[:,1], color=red)
    ax1.tick_params(axis='y', labelcolor=red)
    
    ax2 = ax1.twinx()  
    
    ax2.set_ylabel(y2, color=blue)
    if smooth:
        ax2.plot(cel_ar[:,0], cel_ar[:,1],'.' ,color=blue)
        ax2.plot(cel_ar[:,0],smooth_cel_ar*1000, color=blue)
    else:
        ax2.plot(cel_ar[:,0], cel_ar[:,1] ,color=blue)
    ax2.tick_params(axis='y', labelcolor=blue)
    
    fig.tight_layout()  
    plt.title(f'{material} - MD - {temp}K')
    plt.xlim(eng_ar[0,0], eng_ar[-1,0])
    if save_fig:
        plt.savefig(f'equilibrium_curve_{material}_{temp}K_{y1[0]}_{y2[0]}.jpg',dpi=300,bbox_inches='tight')
    plt.show()

def plot_LV(file_path, num_steps, dist, temp, material, save_fig=False):
    repeat=int(num_steps/dist+2)
    df_LV1=pd.read_csv(file_path,header=None ,sep='\s+').iloc[1:repeat,:2]
    df_LV2=pd.read_csv(file_path,header=None ,sep='\s+').iloc[repeat+1:2*repeat,:2]
    df_LV3=pd.read_csv(file_path,header=None ,sep='\s+').iloc[2*repeat+1:3*repeat,:2]
    
    LV1=df_LV1.to_numpy(dtype=float)
    LV2=df_LV2.to_numpy(dtype=float)
    LV3=df_LV3.to_numpy(dtype=float)
    
    plt.plot(LV1[:,0],LV1[:,1] ,label='Lattice Length 1',c=red)
    plt.plot(LV2[:,0],LV2[:,1], label='Lattice Length 2',c=blue)
    plt.plot(LV3[:,0],LV3[:,1], label='Lattice Length 3',c='k')
    plt.xlabel('Frame Number')
    plt.ylabel('Length (Å)')
    plt.legend()
    plt.title(f'{material} - MD - {temp}K')
    plt.xlim(LV1[0,0], LV1[-1,0])
    plt.show()
    
    smooth_LV,R2=smoothing(LV1[:,0],np.sqrt(LV1[:,1]**2+LV2[:,1]**2+LV3[:,1]**2),type_filter='GPR')
    
    plt.plot(LV1[:,0],np.sqrt(LV1[:,1]**2+LV2[:,1]**2+LV3[:,1]**2),'.' ,c=red)
    plt.plot(LV1[:,0],smooth_LV, label=f'Smoothed, R$^2$={np.round(R2,3)}',c=red)
    plt.xlabel('Frame Number')
    plt.ylabel('Lattice Vector (Å)')
    plt.title(f'{material} - MD - {temp}K')
    plt.xlim(LV1[0,0], LV1[-1,0])
    plt.legend()
    if save_fig:
        plt.savefig(f'Lattice_Vector_Curve_{material}_{temp}K.jpg',dpi=300,bbox_inches='tight')
    plt.show()
    
    return 

def plot_MSD(file_path,res, temp, atoms_list,c_list=[red,blue,'k'], save_fig=False):
    for i in range(len(atoms_list)):
        df_MSD=pd.read_csv(file_path,header=None ,sep='\s+').iloc[1+i*res:(i+1)*res,:2]
        x=df_MSD.iloc[:,0].to_numpy(dtype=float)
        MSD=df_MSD.iloc[:,1].to_numpy(dtype=float)*(0.529177)**2  
        plt.plot(x,MSD,label=f'Coordinates for {atoms_list[i]}', c=c_list[i])
        
    plt.xlabel('Time (fs)')
    plt.ylabel(r'r$^2$ (Å$^2$)')
    plt.title(f'MSD - {atoms_list[0]}, {atoms_list[1]} and {atoms_list[2]} in Perovskite - MD - {temp}K')
    plt.legend()
    plt.xlim(x[0],x[-1])
    if save_fig:
        plt.savefig(f'MSD_{atoms_list[0]}{atoms_list[1]}{atoms_list[2]}3_{temp}K.jpg',dpi=300,bbox_inches='tight')
    plt.show()

def plot_RDF(file_path,res, temp, atoms_list,c_list=[red,blue,'k','g','orange','mediumorchid'], save_fig=False):
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
    plt.title(f'RDF - {atoms_list[0]}, {atoms_list[1]} and {atoms_list[2]} in Perovskite - MD - {temp}K')
    plt.legend()
    plt.xlim(r[0],r[-1])
    plt.ylim(-1,17)
    if save_fig:
        plt.savefig(f'RDF_{atoms_list[0]}{atoms_list[1]}{atoms_list[2]}3_{temp}K.jpg',dpi=300,bbox_inches='tight')
    plt.show()

#equilibrium    
plot_equilib('Energy_CV_100K_equilibrium_run.dat', num_steps, dist, 100, 'Perovskite',smooth=False, save_fig=True)
plot_equilib('Energy_CV_300K_equilibrium.dat', num_steps, dist, 300, 'Perovskite',smooth=False, save_fig=True)
plot_equilib('Energy_CV_700K_equilibrium_run.dat', num_steps, dist, 700, 'Perovskite',smooth=False, save_fig=True)

#Temperature steps
steps=150000
d=1000
# Testing Filters
tesing_filters('Lattice_Vector_100_700K.dat', steps, d, save_fig=True)
tesing_filters('Lattice_Vector_700_to_100K.dat', steps, d, save_fig=True)

# E and Temp
plot_equilib('Energy_Temperature_100_700K.dat', steps, d, '100-700', 'Perovskites', y1='Energy (eV)', y2='Temperature (K)', save_fig=True)
plot_equilib('Energy_Temperature_700_to_100K.dat', steps, d, '700-100', 'Perovskites', y1='Energy (eV)', y2='Temperature (K)', save_fig=True)
# E and CV
plot_equilib('Energy_CV_100K_to_700K.dat', steps, d, '100-700', 'Perovskites',smooth=True, save_fig=True)
plot_equilib('Energy_CV_700K_to_100K.dat', steps, d, '700-100', 'Perovskites',smooth=True, save_fig=True)
# Lattice Vectors
plot_LV('Lattice_Vector_100_700K.dat', steps, d, '100-700', 'perovskites', save_fig=True)
plot_LV('Lattice_Vector_700_to_100K.dat', steps, d, '700-100', 'perovskites', save_fig=True)
    

# MSD
plot_MSD('MSD_300K.dat', 51, 300, ['Cs','Pb', 'I'], save_fig=True)
plot_MSD('MSD_700K.dat', 51, 700, ['Cs','Pb', 'I'], save_fig=True)

# DC
atom_list=['Cs','Pb', 'I']
DC_300K=[5.51467e-12,-6.58963e-13,4.56768e-12] #m2/s
DC_700K=[2.45802e-12,-1.86433e-12,2.01744e-11] #m2/s

plt.plot(atom_list, DC_300K,'o', label='300K',c=red)
plt.plot(atom_list,DC_700K, 'o', label='700K',c=blue)
plt.ylabel(r'Diffusion Coefficient (m$^2$/s)')
plt.legend()
plt.savefig('Diffucsion_Coefficients_Diff_temps.jpg',dpi=300,bbox_inches='tight')
plt.show()

# RDF
plot_RDF('RDF_300K.dat', 1001, 300, ['Cs','Pb', 'I'], save_fig=True)
plot_RDF('RDF_700K.dat', 1001, 700, ['Cs','Pb', 'I'], save_fig=True)
