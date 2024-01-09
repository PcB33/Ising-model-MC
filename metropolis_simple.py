# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random as ran
import time as ti

def simulation(l,N,T,J,kb,T_init):
    
    #create 2D Ising model
    Ising=np.empty((l,l),dtype=int)
    
    if(T_init==0):
        #initialize Ising with all spins +1 -> T=0
        for i in range(l):
            for j in range(l):
                Ising[i,j]=1
            
    if(T_init==100):       
        #initialize Ising with all spins random -> T=inf       
        for i in range(l):
            for j in range(l):
                r=ran.random()
                if r<0.5:
                    Ising[i,j]=1
                else:
                    Ising[i,j]=-1
    
            
    #input: coordinates of spin_k (not in array)   
    #output: array with values +-1 of all four neighbors, including boundary conditions
    def nearest_neighbors(i,j):
        left=Ising[i,(j-1)%l]
        right=Ising[i,(j+1)%l]
        up=Ising[(i-1)%l,j]
        down=Ising[(i+1)%l,j]
        
        return np.array([left,right,up,down])
    
    
    #set initial energy and magnetisation
    sum_nn_pairs=0
    for i in range(l):
        for j in range(l):
            nn=nearest_neighbors(i,j)
            for k in range(4):
                sum_nn_pairs+=Ising[i,j]*nn[k]
    sum_nn_pairs*=1/2
    E_0=-J*sum_nn_pairs
    E=E_0
    
    M_0=np.sum(Ising)
    M=M_0
    
    #performs the flip of spin_k and adjusts energy, magnetisation etc
    def flip(i,j,dE):
        Ising[i,j]*=-1
        spin_k_nu=Ising[i,j]
        nonlocal E
        E+=dE
        nonlocal M
        M+=2*spin_k_nu
        
        return
    
    #performs one step of the metropolis algorithm
    def MC_step():
        #choose random spin k to flip
        i=ran.randint(0,l-1)
        j=ran.randint(0,l-1)
        spin_k_mu=Ising[i,j]
        
        #evaluate E_nu - E_mu according to equation 3.10
        spin_i_mu=nearest_neighbors(i,j)
        sum_nn=np.sum(spin_i_mu)
        dE=2*J*sum_nn*spin_k_mu
        
        #calculate the acceptance ratio and flip spin_k accordingly
        if dE<=0:
            flip(i,j,dE)
        else:
            A=np.exp(-beta*dE)
            r=ran.random()
            if r<A:
                flip(i,j,dE)
        
        return
    
    #perform N steps
    for i in range(N*l**2):
        MC_step()
    
    
    #create image
    plt.matshow(Ising,cmap=ListedColormap(['k','w']))
    plt.show()
    
    print('E:',E)
    print('M:',M)
    
    return

#Define variables
J=1
kb=1
T_init=0
T=1.8
beta=1/(kb*T)
N=100 #number of sweeps
l=10 #dimension of Ising model

t1=ti.time()
simulation(l,N,T,J,kb,T_init)
t2=ti.time()
print('computation time',t2-t1)    