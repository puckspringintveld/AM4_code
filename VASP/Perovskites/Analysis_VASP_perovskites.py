# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:07:58 2025

@author: 20211382
"""

import numpy as np
import matplotlib.pyplot as plt
red = 'tab:red'

### CsPbI3
# R1
R1_opt=np.genfromtxt('CsPbI3/R1/G_OPT_ISIF_7.dat')

plt.plot(R1_opt[:,0], R1_opt[:,2],'--*', c=red, label='Converged LV = 6.4 Å')
plt.xlabel('Lattice Vector Step')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()

#R2
atoms_list=['Pb', 'Cs', 'I']
e_levels=['s', 'p', 'd']
s=np.genfromtxt('CsPbI3/R2/s301.dat')
p=np.genfromtxt('CsPbI3/R2/p301.dat')
d=np.genfromtxt('CsPbI3/R2/d301.dat')

list_e_levels=[s,p,d]

for e_level in range(3):
    for i,atom in enumerate(atoms_list):
        if atom=='I':
            x=list_e_levels[e_level][:,0]
            y=list_e_levels[e_level][:,i+1]
            y2=list_e_levels[e_level][:,i+2]
            y3=list_e_levels[e_level][:,i+3]
            plt.plot(x,y+y2+y3,label=f'{atom} ({e_levels[e_level]})')
        else:
            x=list_e_levels[e_level][:,0]
            y=list_e_levels[e_level][:,i+1]
            plt.plot(x,y,label=f'{atom} ({e_levels[e_level]})')
        
plt.legend()
plt.xlim(x.min(),x.max())
plt.xlabel('Energy (eV)')
plt.ylabel('DOS')
plt.xlim(-6,6)
plt.savefig('DOS_CsPbI3.jpg', dpi=300,bbox_inches='tight')
plt.show()

#Calculate Bandgap
# we know I(p) and Pb(p) cause it
x=p[:,0]
Pb_p=p[:,1]
I_p=p[:,3]+p[:,4]+p[:,5]
#extract manually
band_gap_DOS_CsPbI3=0.879 + 0.294


#R3
band=np.genfromtxt('CsPbI3/R3/band.dat')

above_F=[]
bellow_F=[]
for i in range(1,len(band[0,:])):
    plt.plot(band[:,0],band[:,i], c='k', linewidth=0.75, alpha=0.9) 
    mini=np.where(band[:,i]==band[:,i].min())
    maxi=np.where(band[:,i]==band[:,i].max())
    if band[:,i][mini[0]][0]>0:
        above_F.append([mini[0][0],band[:,0][mini[0]][0],band[:,i][mini[0]][0]])
    elif band[:,i][maxi[0]][0]<0:
        bellow_F.append([maxi[0][0],band[:,0][maxi[0]][0],band[:,i][maxi[0]][0]])
above_F=np.array(above_F,dtype=float)
bellow_F=np.array(bellow_F,dtype=float)
plt.xlim(band[:,0].min(),band[:,0].max())
minimum_above_F=np.where(above_F[:,2]==above_F[:,2].min())
maximum_bellow_F=np.where(bellow_F[:,2]==bellow_F[:,2].max())
plt.plot(above_F[:,1][minimum_above_F],above_F[:,2][minimum_above_F], '*',c=red)
band_gap=above_F[:,2][minimum_above_F]-bellow_F[:,2][maximum_bellow_F]
plt.plot(bellow_F[:,1][maximum_bellow_F],bellow_F[:,2][maximum_bellow_F], '*', c=red, label=f'Bandgap = {np.round(band_gap[0],3)} eV')
plt.xlabel('K point')
plt.ylabel('Energy (eV)')
plt.legend()
plt.ylim(-6,6)
plt.savefig('Band_Structure_CsPbI3.jpg', dpi=300,bbox_inches='tight')
plt.show()

### MAPbI3
# R1
R1_opt_MA=np.genfromtxt('MAPbI3/G_OPT_ISIF_2_R1.dat')

plt.plot(R1_opt_MA[:,0], R1_opt_MA[:,2],'--*', c=red, label='Converged LV = 6.4 Å')
plt.xlabel('Ionic Vector Step')
plt.ylabel('Energy (eV)')
plt.legend()
plt.title('Optimization MAPbI3 ISIF=2')
plt.show()

#R2
atoms_list_MA=['C', 'N', 'H', 'Pb', 'I']
s_MA=np.genfromtxt('MAPbI3/R2/s301.dat')
p_MA=np.genfromtxt('MAPbI3/R2/p301.dat')
d_MA=np.genfromtxt('MAPbI3/R2/d301.dat')

list_e_levels=[s_MA,p_MA,d_MA]

for e_level in range(3):
    for i,atom in enumerate(atoms_list_MA):
        if atom=='I':
            x=list_e_levels[e_level][:,0]
            y=list_e_levels[e_level][:,i+6]
            y2=list_e_levels[e_level][:,i+7]
            y3=list_e_levels[e_level][:,i+8]
            plt.plot(x,y+y2+y3,label=f'{atom} ({e_levels[e_level]})')
        elif atom=='H':
            x=list_e_levels[e_level][:,0]
            y=list_e_levels[e_level][:,i+1]
            y2=list_e_levels[e_level][:,i+2]
            y3=list_e_levels[e_level][:,i+3]
            plt.plot(x,y+y2+y3,label=f'C{atom} ({e_levels[e_level]})')
            y4=list_e_levels[e_level][:,i+4]
            y5=list_e_levels[e_level][:,i+5]
            y6=list_e_levels[e_level][:,i+6]
            plt.plot(x,y4+y5+y6,label=f'N{atom} ({e_levels[e_level]})')
        elif atom=='Pb':
            x=list_e_levels[e_level][:,0]
            y=list_e_levels[e_level][:,i+6]
            plt.plot(x,y,label=f'{atom} ({e_levels[e_level]})')
        else:
            x=list_e_levels[e_level][:,0]
            y=list_e_levels[e_level][:,i+1]
            plt.plot(x,y,label=f'{atom} ({e_levels[e_level]})')
        
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.xlim(x.min(),x.max())
plt.xlabel('Energy (eV)')
plt.ylabel('DOS')
plt.title('MAPbI3 DOS')
plt.xlim(-6,6)
plt.savefig('DOS_MAPbI3.jpg', dpi=300,bbox_inches='tight')
plt.show()

#Calculate Bandgap
#between I(p) and Pb(p)
x_p=p_MA[:,0]
I_p=p_MA[:,10]+p_MA[:,11]+p_MA[:,12]
Pb_p=p_MA[:,9]
#extract manually
band_gap_DOS_MAPbI3=1.18 + 0.394


#R3
band_MA=np.genfromtxt('MAPbI3/R3/band.dat')

above_F=[]
bellow_F=[]
for i in range(1,len(band_MA[0,:])):
    plt.plot(band_MA[:,0],band_MA[:,i], c='k', linewidth=0.75, alpha=0.9) 
    mini=np.where(band_MA[:,i]==band_MA[:,i].min())
    maxi=np.where(band_MA[:,i]==band_MA[:,i].max())
    if band_MA[:,i][mini[0]][0]>0:
        above_F.append([mini[0][0],band_MA[:,0][mini[0]][0],band_MA[:,i][mini[0]][0]])
    elif band_MA[:,i][maxi[0]][0]<0:
        bellow_F.append([maxi[0][0],band_MA[:,0][maxi[0]][0],band_MA[:,i][maxi[0]][0]])
above_F=np.array(above_F,dtype=float)
bellow_F=np.array(bellow_F,dtype=float)
plt.xlim(band_MA[:,0].min(),band_MA[:,0].max())
minimum_above_F=np.where(above_F[:,2]==above_F[:,2].min())
maximum_bellow_F=np.where(bellow_F[:,2]==bellow_F[:,2].max())
plt.plot(above_F[:,1][minimum_above_F],above_F[:,2][minimum_above_F], '*',c=red)
band_gap_MA=above_F[:,2][minimum_above_F]-bellow_F[:,2][maximum_bellow_F]
plt.plot(bellow_F[:,1][maximum_bellow_F],bellow_F[:,2][maximum_bellow_F], '*', c=red, label=f'Bandgap = {np.round(band_gap_MA[0],3)} eV')
plt.xlabel('K point')
plt.ylabel('Energy (eV)')
plt.legend()
plt.ylim(-6,6)
plt.title('MAPbI3 Band Structure')
plt.savefig('Band_Structure_MAPbI3.jpg', dpi=300,bbox_inches='tight')
plt.show()

print(band_gap_DOS_CsPbI3,band_gap_DOS_MAPbI3)

### CsPbBr3
#R2
atoms_list_Br=['Pb', 'Cs', 'Br']
s_Br=np.genfromtxt('CsPbBr3/R2/s301.dat')
p_Br=np.genfromtxt('CsPbBr3/R2/p301.dat')
d_Br=np.genfromtxt('CsPbBr3/R2/d301.dat')

list_e_levels=[s_Br,p_Br,d_Br]

for e_level in range(3):
    for i,atom in enumerate(atoms_list_Br):
        if atom=='Br':
            x=list_e_levels[e_level][:,0]
            y=list_e_levels[e_level][:,i+1]
            y2=list_e_levels[e_level][:,i+2]
            y3=list_e_levels[e_level][:,i+3]
            plt.plot(x,y+y2+y3,label=f'{atom} ({e_levels[e_level]})')
        else:
            x=list_e_levels[e_level][:,0]
            y=list_e_levels[e_level][:,i+1]
            plt.plot(x,y,label=f'{atom} ({e_levels[e_level]})')
        
plt.legend()
plt.xlim(x.min(),x.max())
plt.xlabel('Energy (eV)')
plt.ylabel('DOS')
plt.xlim(-6,6)
plt.savefig('DOS_CsPbBr3.jpg', dpi=300,bbox_inches='tight')
plt.show()

#Calculate Bandgap
#between Br(p) and Pb(p)
x_p=p_Br[:,0]
Br_p=p_Br[:,3]+p_Br[:,4]+p_Br[:,5]
Pb_p_2=p_Br[:,1]
#extract manually
band_gap_DOS_CsPbBr3=1.18 + 0.693



#R3
band_Br=np.genfromtxt('CsPbBr3/R3/band.dat')

above_F=[]
bellow_F=[]
for i in range(1,len(band_Br[0,:])):
    plt.plot(band_Br[:,0],band_Br[:,i], c='k', linewidth=0.75, alpha=0.9) 
    mini=np.where(band_Br[:,i]==band_Br[:,i].min())
    maxi=np.where(band_Br[:,i]==band_Br[:,i].max())
    if band_Br[:,i][mini[0]][0]>0:
        above_F.append([mini[0][0],band_Br[:,0][mini[0]][0],band_Br[:,i][mini[0]][0]])
    elif band_Br[:,i][maxi[0]][0]<0:
        bellow_F.append([maxi[0][0],band_Br[:,0][maxi[0]][0],band_Br[:,i][maxi[0]][0]])
above_F=np.array(above_F,dtype=float)
bellow_F=np.array(bellow_F,dtype=float)
plt.xlim(band_Br[:,0].min(),band_Br[:,0].max())
minimum_above_F=np.where(above_F[:,2]==above_F[:,2].min())
maximum_bellow_F=np.where(bellow_F[:,2]==bellow_F[:,2].max())
plt.plot(above_F[:,1][minimum_above_F],above_F[:,2][minimum_above_F], '*',c=red)
band_gap_Br=above_F[:,2][minimum_above_F]-bellow_F[:,2][maximum_bellow_F]
plt.plot(bellow_F[:,1][maximum_bellow_F],bellow_F[:,2][maximum_bellow_F], '*', c=red, label=f'Bandgap = {np.round(band_gap_Br[0],3)} eV')
plt.xlabel('K point')
plt.ylabel('Energy (eV)')
plt.legend()
plt.ylim(-6,6)
plt.savefig('Band_Structure_CsPbBr3.jpg', dpi=300,bbox_inches='tight')
plt.show()