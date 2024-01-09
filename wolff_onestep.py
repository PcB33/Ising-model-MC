# -*- coding: utf-8 -*-
import numpy as np
import random as ran

#input: coordinates of new spin and cluster
#output: True if new spinis already in cluster, False otherwise
def is_in_cluster(i,j,cluster):
    
    for k in range(int(cluster.size/2)):
        i_k=cluster[k,0]
        j_k=cluster[k,1]
        
        if (i==i_k and j==j_k):
            return True
    
    return False

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
        print('already part of cluster')
        return cluster
    
    #checks if spins are same. If not, return original cluster
    if (Ising[i,j]!=Ising[i_orig,j_orig]):
        print('spins non equal')
        global n
        n+=1
        return cluster
    
    #if spin does not pass P_add, return original cluster
    r=ran.random()
    if (r>P_add):
        print('P-fail')
        global m
        m+=1
        return cluster
    
    new_spin=np.array([np.array([i,j])])
    new_cluster=np.append(cluster,new_spin,axis=0)
    print('spin added')
    print('nc:',new_cluster)
    return new_cluster

#input: coordinates of a spin (not in array) and existing, non-empty cluster
#effect: adds neighboring spins of same orientation at probability P_add, if not already part of cluster
def check_neighbors(i,j,cluster):
    new_cluster=check_spin(i,j,i,(j-1)%l,cluster)#left
    print('left complete')
    new_cluster=check_spin(i,j,i,(j+1)%l,new_cluster)#right
    print('right complete')
    new_cluster=check_spin(i,j,(i-1)%l,j,new_cluster)#up
    print('up complete')
    new_cluster=check_spin(i,j,(i+1)%l,j,new_cluster)#down
    print('down complete')
    
    return new_cluster

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


#choose random spin as seed
i=ran.randint(0,l-1)
j=ran.randint(0,l-1)
spin_seed_coord=np.array([i,j])

n=0
m=0

#define P_add
#P_add=1-np.exp(-2*beta*J)
P_add=0.4

#define the original cluster with just the seed spin
cluster=np.array([spin_seed_coord])

print('original cluster:',cluster)

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
        print('k:',k)
        #coordinates of the new k-th seed
        k_i=cluster[cluster_size_new-1-k,0]
        k_j=cluster[cluster_size_new-1-k,1]
        print('new seed:',np.array([k_i,k_j]))
        #this is where the neighboring spins of the seed are evaluated and the cluster
        ##is adjusted accordingly
        bigger_cluster=check_neighbors(k_i,k_j,cluster_k)
        print('bigger_cluster',bigger_cluster)
        cluster_k=bigger_cluster
        print('---')
    #check if the cluster grew larger during the last for loop. If not, the cluster is complete
    if (cluster_k.size == cluster.size):
        cluster_complete=True
        final_cluster=cluster_k
        print('cluster complete: True')
    cluster_size_old=cluster_size_new
    cluster_size_new=int(cluster_k.size/2)
    print('cluster growth:',cluster_size_new-cluster_size_old)
    cluster=cluster_k
    print('cluster at end of pass:',cluster)
    print('-------------------')

print('final cluster:',final_cluster)
print('m:',m)
print('n:',n)
