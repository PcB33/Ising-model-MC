# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random as ran
import time as ti

def simulation(l,N,T,J,kb,Ising,num):
    
    if (num==1 or num==2 or num==3 or num==4 or num==5 or num==10 or num==50 or num==100 or num==500 or num==1000 or num==1500):
        #create image
        plt.matshow(Ising,cmap=ListedColormap(['k','w']))
        plt.show()
        print(num,' sweeps')
            
    #input: coordinates of spin_k (not in array)   
    #output: array with values +-1 of all four neighbors, including boundary conditions
    def nearest_neighbors(i,j):
        left=Ising[i,(j-1)%l]
        right=Ising[i,(j+1)%l]
        up=Ising[(i-1)%l,j]
        down=Ising[(i+1)%l,j]
        
        return np.array([left,right,up,down])
    
    
    #set initial energy, magnetisation, etc
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
    
    '''
    print('E:',E)
    print('M:',M)
    '''
    return Ising, E, M

#Define variables
J=1
kb=1
T_init=0
T=2.27
beta=1/(kb*T)
block_size=100 #for blocking method
N=100 #number of sweeps
eq=0 # point where equilibrium is assumed to have been reached
l=50 #dimension of Ising model

#create 2D Ising model
Ising=np.empty((l,l),dtype=int)


if (T_init==0):
    #initialize Ising with all spins +1 -> T=0
    for i in range(l):
        for j in range(l):
            Ising[i,j]=1
    #create initial image
    plt.matshow(Ising,cmap=ListedColormap(['w','w']))
    plt.show()
    print('initial state')

if (T_init==100):
    #initialize Ising with all spins random -> T=inf       
    for i in range(l):
        for j in range(l):
            r=ran.random()
            if r<0.5:
                Ising[i,j]=1
            else:
                Ising[i,j]=-1
    #create initial image
    plt.matshow(Ising,cmap=ListedColormap(['k','w']))
    plt.show()
    print('initial state')        

      
#calculate energies after all steps
t1=ti.time()
U=np.empty(N)
Mag=np.empty(N)
for i in range(N):
    Ising,U[i],Mag[i]=simulation(l,1,T,J,kb,Ising,i)
t2=ti.time()

#print final Ising, energy and magnetisation
plt.matshow(Ising,cmap=ListedColormap(['k','w']))
plt.show()
print(N,' sweeps')


print('final energy',U[N-1])
print('final magnetisation',Mag[N-1])


#energy and magnetisation curves
N_var=np.linspace(0,N,N)
plt.plot(N_var,U)
plt.xlabel('Number of sweeps')
plt.ylabel('Internal energy')
plt.show()
plt.plot(N_var,Mag)
plt.xlabel('Number of sweeps')
plt.ylabel('Magnetisation')
plt.show()


#calculation of mean and std for energy and magnetisation
U_mean=np.mean(U[eq:])
U_std=np.std(U[eq:])
Mag_mean=np.mean(Mag[eq:])
Mag_std=np.std(Mag[eq:])

print('U_mean:',U_mean,'+/-',U_std)
print('Mag_mean:',Mag_mean,'+/-',Mag_std)

#calculation of specific heat including std using blocking method
U_sq=U**2
c_blocks=np.empty(int((N-eq)/block_size))
for i in range(c_blocks.size):
    U_mean_block=np.mean(U[(eq+i*block_size):(eq+(i+1)*block_size)])
    U_sq_mean_block=np.mean(U_sq[(eq+i*block_size):(eq+(i+1)*block_size)])
    c_blocks[i]=beta**2*(U_sq_mean_block-U_mean_block**2)
    
print('c_mean:',np.mean(c_blocks),'+/-',np.std(c_blocks))

#calculation of total specific heat capacity (not per site) according to equation 3.15
U_sq_mean=np.mean(U_sq[eq:])
c=beta**2*(U_sq_mean-U_mean**2)
print('c:',c)



#calculation of autocorrelation function of m according to equation 3.21
chi_m=np.zeros_like(Mag)
for i in range(chi_m.size):
    chi_m[i]=1/(N-i)*np.sum(Mag[j]*Mag[j+i] for j in range(0,N-i))-1/(N-i)*np.sum(Mag[j] for j in range(0,N-i))*1/(N-i)*np.sum(Mag[j+i] for j in range(0,N-i))

#calculation of integrated corellation time according to equation 3.20
tau_m=1/chi_m[0]*np.sum(chi_m[j] for j in range(0,int(0.5*N)))
print('tau_m:',tau_m)

plt.plot(N_var,chi_m/chi_m[0])
plt.xlabel('Number of sweeps')
plt.ylabel('$\\chi_m$')
plt.show()

#calculation of autocorrelation function of u according to equation 3.21
chi_u=np.zeros_like(U)
for i in range(chi_u.size):
    chi_u[i]=1/(N-i)*np.sum(U[j]*U[j+i] for j in range(0,N-i))-1/(N-i)*np.sum(U[j] for j in range(0,N-i))*1/(N-i)*np.sum(U[j+i] for j in range(0,N-i))

#calculation of integrated corellation time according to equation 3.20
tau_u=1/chi_u[0]*np.sum(chi_u[j] for j in range(0,int(0.8*N)))
print('tau_u:',tau_u)


plt.plot(N_var,chi_u/chi_u[0])
plt.xlabel('Number of sweeps')
plt.ylabel('$\\chi_u$')
plt.show()

'''
plt.loglog(N_var,chi_u/chi_u[0])
plt.show()

plt.loglog(N_var[:70],chi_u[:70]/chi_u[0])
plt.show()
'''

#print variables and time
print('total time:',t2-t1)
print('l:',l)
print('N:',N)
print('T:',T)


#times: for l=50, N=100: 4.5s. scales as l^2 and N
#remark: if you start from T=inf, then the Ising board is clustered before its in equilibrium
#equilibrium times: l=50 -> N = 1800. l=100 -> N = 3000 (starting from T=inf at T=2)   