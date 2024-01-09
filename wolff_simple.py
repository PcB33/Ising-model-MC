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
    
    #input: coordinates of spin_k (not in array) 
    #returns array with coordinates [i,j] of the four neighboring spins
    def nearest_neighbors_coord(i,j):
        left=np.array([i,(j-1)%l])
        right=np.array([i,(j+1)%l])
        up=np.array([(i-1)%l,j])
        down=np.array([(i+1)%l,j])
        
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
    #print(E)
    
    M_0=np.sum(Ising)
    M=M_0
    #print(M)
    
    
    #performs the flip of spin with coordinates (i,j) and adjusts energy, magnetisation etc
    def flip(i,j):
        Ising[i,j]*=-1
        
        return
    
    #adjusts E and M of the Ising board after 
    def adjust_EM(m,n,cluster_size,spin_k_mu):
        dE=2*J*(m-n)
        dM=2*cluster_size*spin_k_mu*(-1)
        nonlocal E
        E+=dE
        nonlocal M
        M+=dM
        #print('cluster size:',cluster_size)
        #print('m:',m)
        #print('n:',n)
        #print('dE:',dE)
        #print('dM:',dM)
        
        return
    
    #performs one step of the wolff algorithm
    def MC_step():
        #input: coordinates of new spin and cluster
        #output: True if new spinis already in cluster, False otherwise
        def is_in_cluster(i,j,cluster):
            
            for k in range(int(cluster.size/2)):
                i_k=cluster[k,0]
                j_k=cluster[k,1]
                
                if (i==i_k and j==j_k):
                    return True
            
            return False
        
        def check_rejected(i,j):
            for k in range(int(rejected.size/2)):
                if (i==rejected[2*k] and j==rejected[2*k+1]):
                    nonlocal m
                    m-=1
            
            return
        
        #input: coordinates of original spin as well as coordinates of spin which is to be evaluated along with cluster
        #effect: returns original cluster if 
        #a) spin already part of cluster
        #b) spins are not same orientation
        #c) spin does not pass P_add
        #if none of these happen, new_cluster with added spin coordinates is returned
        def check_spin(i_orig,j_orig,i,j,cluster):
            
            #checks if (i,j) is already part of cluster. If so, return original cluster
            is_part=is_in_cluster(i,j,cluster)
            if (is_part==True):
                #print('already part of cluster')
                #nonlocal already_part
                #already_part+=1
                return cluster
            
            #checks if spins are same. If not, return original cluster
            if (Ising[i,j]!=Ising[i_orig,j_orig]):
                #print('spins non equal')
                nonlocal n
                n+=1
                return cluster
            
            #if spin does not pass P_add, return original cluster
            r=ran.random()
            if (r>P_add):
                #print('P-fail')
                nonlocal m
                m+=1
                nonlocal rejected
                rejected=np.append(rejected,i)
                rejected=np.append(rejected,j)
                #print('rejected:',rejected)
                return cluster
            
            new_spin=np.array([np.array([i,j])])
            new_cluster=np.append(cluster,new_spin,axis=0)
            check_rejected(i,j)
            #print('spin added')
            #print('nc:',new_cluster)
            return new_cluster
        
        #input: coordinates of a spin (not in array) and existing, non-empty cluster
        #effect: adds neighboring spins of same orientation at probability P_add, if not already part of cluster
        def check_neighbors(i,j,cluster):
            new_cluster=check_spin(i,j,i,(j-1)%l,cluster)#left
            #print('left complete')
            new_cluster=check_spin(i,j,i,(j+1)%l,new_cluster)#right
            #print('right complete')
            new_cluster=check_spin(i,j,(i-1)%l,j,new_cluster)#up
            #print('up complete')
            new_cluster=check_spin(i,j,(i+1)%l,j,new_cluster)#down
            #print('down complete')
            
            return new_cluster
        
        '''
        #define variables
        l=10
        beta=1
        J=1
        
        
        #create 2D Ising model
        Ising=np.empty((l,l),dtype=int)
        
        #initialize Ising with all spins random -> T=inf       
        for i in range(l):
            for j in range(l):
                r=ran.random()
                if r<0.5:
                    Ising[i,j]=1
                else:
                    Ising[i,j]=-1
        '''
        
        #choose random spin as seed
        i=ran.randint(0,l-1)
        j=ran.randint(0,l-1)
        spin_seed=Ising[i,j]
        spin_seed_coord=np.array([i,j])
        
        #define n and already_part variables
        n=0
        m=0
        rejected=np.empty(0)
        #already_part=0
        #P_fails=0
        
        #define P_add
        P_add=1-np.exp(-2*beta*J)
        
        #define the original cluster with just the seed spin
        cluster=np.array([spin_seed_coord])
        
        #print('original cluster:',cluster)
        
        #auxiliary variables
        cluster_complete = False
        cluster_size_old=0
        cluster_size_new=1
        
        #Build the cluster. The while loop continues as long as at least one spin
        ##was added to the cluster during the last loop
        while(cluster_complete==False):
            cluster_k=cluster
            #for loop for all the new seeds that were added to the cluster during the last while loop
            for k in range (cluster_size_new-cluster_size_old):
                #print('k:',k)
                #coordinates of the new k-th seed
                k_i=cluster[cluster_size_new-1-k,0]
                k_j=cluster[cluster_size_new-1-k,1]
                #print('new seed:',np.array([k_i,k_j]))
                #this is where the neighboring spins of the seed are evaluated and the cluster
                ##is adjusted accordingly
                bigger_cluster=check_neighbors(k_i,k_j,cluster_k)
                #print('bigger_cluster',bigger_cluster)
                cluster_k=bigger_cluster
                #print('---')
            #check if the cluster grew larger during the last for loop. If not, the cluster is complete
            if (cluster_k.size == cluster.size):
                cluster_complete=True
                final_cluster=cluster_k
                #print('cluster complete: True')
            cluster_size_old=cluster_size_new
            cluster_size_new=int(cluster_k.size/2)
            #print('cluster growth:',cluster_size_new-cluster_size_old)
            cluster=cluster_k
            #print('cluster at end of pass:',cluster)
            #print('-------------------')
        
        #print('final cluster:',final_cluster)
        
        #flip the orientations of all the spins in the final cluster
        for k in range(int(final_cluster.size/2)):
            k_i=final_cluster[k,0]
            k_j=final_cluster[k,1]
            flip(k_i,k_j)
        
        #m=4*final_cluster.size/2-n-already_part-final_cluster.size/2
        adjust_EM(m,n,final_cluster.size/2,spin_seed)
        #create image
        '''
        plt.matshow(Ising,cmap=ListedColormap(['k','w']))
        plt.show()
        print('E:',E)
        print('M:',M)
        '''
        
        return
    
    
    
    #perform N steps
    for i in range(N):
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
T=20
beta=1/(kb*T)
N=100 #number of steps (single steps, not sweeps)
l=10 #dimension of Ising model

t1=ti.time()
simulation(l,N,T,J,kb,T_init)
t2=ti.time()
print('computation time',t2-t1)  