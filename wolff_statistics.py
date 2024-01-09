# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random as ran
import time as ti
from scipy.integrate import quad

def statistics(l,N,T,J,kb,eq,block_size,beta,T_init):
    
    def simulation(l,N,T,J,kb,Ising):
                
        #input: coordinates of spin_k (not in array)   
        #output: array with values +-1 of all four neighbors, including boundary conditions
        def nearest_neighbors(i,j):
            left=Ising[i,(j-1)%l]
            right=Ising[i,(j+1)%l]
            up=Ising[(i-1)%l,j]
            down=Ising[(i+1)%l,j]
            
            return np.array([left,right,up,down])
        
        
        #set initial energy, magnetisation
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
        
        #performs one step of the metropolis algorithm
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
    
            
            return
        
        #perform N steps
        for i in range(N*l**2):
            MC_step()
        
        '''
        #create image
        plt.matshow(Ising,cmap=ListedColormap(['k','w']))
        plt.show()
        '''
        
        '''
        print('E:',E)
        print('M:',M)
        '''
        return Ising, E, M
    
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
           
    #calculate energies after all steps
    #t1=ti.time()
    U=np.empty(N)
    Mag=np.empty(N)
    for i in range(N):
        Ising,U[i],Mag[i]=simulation(l,1,T,J,kb,Ising)
    #t2=ti.time()
    
    #print final Ising, energy and magnetisation
    plt.matshow(Ising,cmap=ListedColormap(['k','w']))
    plt.show()
    #print('final energy',U[N-1])
    #print('final magnetisation',Mag[N-1])
    
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
    plt.plot(N_var,np.abs(Mag))
    plt.xlabel('Number of sweeps')
    plt.ylabel('abs. val. Magnetisation')
    plt.show()
    
    #calculation of mean and std fo energy and magnetisation
    U_mean=np.mean(U[eq:])
    U_std=np.std(U[eq:])
    Mag_mean=np.mean(Mag[eq:])
    Mag_std=np.std(Mag[eq:])
    print(Mag_mean)
    
    #print('U_mean:',U_mean,'+/-',U_std)
    #print('Mag_mean:',Mag_mean,'+/-',Mag_std)
    
    #calculation of specific heat including std using blocking method
    U_sq=U**2
    c_blocks=np.empty(int((N-eq)/block_size))
    for i in range(c_blocks.size):
        U_mean_block=np.mean(U[(eq+i*block_size):(eq+(i+1)*block_size)])
        U_sq_mean_block=np.mean(U_sq[(eq+i*block_size):(eq+(i+1)*block_size)])
        c_blocks[i]=beta**2*(U_sq_mean_block-U_mean_block**2)
    
    c_mean=np.mean(c_blocks)
    c_std=np.std(c_blocks)
    #print('c_mean:',c_mean,'+/-',c_std)
    
    #calculation of total specific heat capacity (not per site) according to equation 3.15
    #U_sq_mean=np.mean(U_sq[eq:])
    #c=beta**2*(U_sq_mean-U_mean**2)
    #print('c:',c)
    
    #calculation of autocorrelation function of m according to equation 3.21
    chi_m=np.zeros_like(Mag)
    for i in range(chi_m.size):
        chi_m[i]=1/(N-i)*np.sum(Mag[j]*Mag[j+i] for j in range(0,N-i))-1/(N-i)*np.sum(Mag[j] for j in range(0,N-i))*1/(N-i)*np.sum(Mag[j+i] for j in range(0,N-i))
    
    #calculation of integrated corellation time according to equation 3.20
    tau_m=1/chi_m[0]*np.sum(chi_m[j] for j in range(0,int(0.8*N)))
    print('tau_m:',tau_m)
    
    plt.plot(N_var,chi_m/chi_m[0])
    plt.xlabel('Number of sweeps')
    plt.ylabel('$\\chi_m$')
    plt.show()
    
    #print variables and time
    #print('total time:',t2-t1)
    #print('l:',l)
    #print('N:',N)
    #print('T:',T)
    
    
    #times: for l=50, N=100: 4.5s. scales as l^2 and N
    #remark: if you start from T=inf, then the Ising board is clustered before its in equilibrium
    #equilibrium times: l=50 -> N = 800. l=100 -> N = 2000 (starting from T=inf at T=2)
    
    print('T:',T)
    return U_mean,U_std,Mag_mean,Mag_std,c_mean,c_std,tau_m

#define variables
J=1
kb=1
T_c=2.269185 #critical temperature
T_init=100
T_start=1.2
T_end=4
'''
T_step=0.2
T=np.linspace(T_start,T_end,int((T_end-T_start)/T_step+2))
'''
T=np.array([1.2,1.4,1.6,1.8,2.0,2.1,2.2,2.25,2.3,2.4,2.5,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0])
beta=1/(kb*T)
block_size=np.array([5,5,5,5,5,5,10,10,10,10,10,10,10,100,100,100,100,100,100]) #for blocking method
N=np.array([50,55,60,65,100,110,250,300,320,500,1000,1200,2000,5000,5000,5000,5000,5000,5000]) #number of sweeps
eq=np.array([30,35,40,45,80,90,150,170,200,300,500,600,800,2000,2000,2000,2000,2000,2000]) # point where equilibrium is assumed to have been reached
l=10 #dimension of Ising model


U_means=np.empty(int(T.size))
U_stds=np.empty(int(T.size))
Mag_means=np.empty(int(T.size))
Mag_stds=np.empty(int(T.size))
c_means=np.empty(int(T.size))
c_stds=np.empty(int(T.size))
tau_ms=np.empty(int(T.size))


#run simulation at all temperatures
t1=ti.time()
for i in range(T.size):
    t3=ti.time()
    U_means[i],U_stds[i],Mag_means[i],Mag_stds[i],c_means[i],c_stds[i],tau_ms[i]=statistics(l,N[i],T[i],J,kb,eq[i],block_size[i],beta[i],T_init)
    t4=ti.time()
    print('time for run:',t4-t3)
t2=ti.time()


#calculation of entropy
T_forS=np.empty(int(T.size)+1)
T_forS[0]=0.01
for i in range(int(T.size)):
    T_forS[i+1]=T[i]
    
c_forS=np.empty(int(T.size)+1)
c_forS[0]=0
for i in range(int(T.size)):
    c_forS[i+1]=c_means[i]

S=np.empty(int(T_forS.size))
    
for i in range(S.size):
    S[i]=np.trapz(c_forS[:i+1]/T_forS[:i+1],x=T_forS[:i+1])


#exact functions. source: https://journals.aps.org/pr/pdf/10.1103/PhysRev.65.117
T_exact=np.linspace(T_start,T_end,1000)
T_exact_mag=np.linspace(T_start,T_c,1000)
beta_exact=1/(kb*T_exact)
beta_exact_mag=1/(kb*T_exact_mag)
L=beta_exact*J
K=L
k=1/(np.sinh(2*L)*np.sinh(2*K))

U_exact=np.empty(T_exact.size)
def integrand(x,k):
    return 1/np.sqrt(1-4*k*(1+k)**(-2)*np.sin(x)**2)

for i in range(T_exact.size):
    integral=quad(integrand,0,np.pi/2,args=k[i])
    U_exact[i]=-J/np.tanh(2*beta_exact[i]*J)*(1+2/np.pi*(2*np.tanh(2*beta_exact[i]*J)**2-1)*integral[0])
    
c_exact=np.zeros_like(T_exact)
for i in range(T_exact.size-1):
    c_exact[i+1]=(U_exact[i+1]-U_exact[i])/(T_exact[i+1]-T_exact[i])
c_exact[0]=c_exact[1]
Mag_exact=(1-np.sinh(2*beta_exact_mag*J)**(-4))**(1/8)
Mag_exact[-1]=0

T_forS_exact=np.empty(int(T_exact.size)+1)
T_forS_exact[0]=0.01
for i in range(int(T_exact.size)):
    T_forS_exact[i+1]=T_exact[i]
    
c_forS_exact=np.empty(int(T_exact.size)+1)
c_forS_exact[0]=0
for i in range(int(T_exact.size)):
    c_forS_exact[i+1]=c_exact[i]

S_exact=np.empty(int(T_forS_exact.size))
    
for i in range(S_exact.size):
    S_exact[i]=np.trapz(c_forS_exact[:i+1]/T_forS_exact[:i+1],x=T_forS_exact[:i+1])
    

#plot results
plt.errorbar(T,U_means/l**2,yerr=U_stds/l**2,marker='.',color='k',linestyle='none',label='simulation',capsize=2)
#plt.plot(T,U_means/l**2)
plt.plot(T_exact,U_exact,label='exact solution')
plt.xlabel('T')
plt.ylabel('u')
plt.title('internal energy per spin')
plt.legend(loc='best')
plt.show()

'''
plt.plot(T_exact,U_exact)
plt.title('exact solution for u')
plt.xlabel('T')
plt.ylabel('u')
plt.legend(loc='best')
plt.show()
'''

plt.plot(T,np.abs(Mag_means)/l**2,'^',color='k',label='simulation')
#plt.errorbar(T,np.abs(Mag_means)/l**2,Mag_stds/l**2,label='simulation')
plt.plot(T_exact_mag,Mag_exact,label='exact solution')
plt.xlabel('T')
plt.ylabel('$\mid m \mid$')
plt.title('magnetisation per spin')
plt.legend(loc='best')
plt.show()
'''
plt.plot(T_exact_mag,Mag_exact)
plt.xlabel('T')
plt.ylabel('m')
plt.title('exact solution for m')
plt.legend(loc='best')
plt.show()
'''

plt.errorbar(T,c_means/l**2,c_stds/l**2,marker='.',color='k',linestyle='none',label='simulation',capsize=2)
#plt.plot(T,c_means/l**2)
plt.plot(T_exact,c_exact,label='exact solution')
plt.xlabel('T')
plt.ylabel('c')
plt.ylim(top=3)
plt.title('specific heat per spin')
plt.legend(loc='best')
plt.show()

plt.errorbar(T,c_means/l**2,c_stds/l**2,marker='.',color='k',label='simulation',capsize=2)
#plt.plot(T,c_means/l**2)
plt.plot(T_exact,c_exact,label='exact solution')
plt.xlabel('T')
plt.ylabel('c')
plt.ylim(top=3)
plt.title('specific heat per spin')
plt.legend(loc='best')
plt.show()

'''
plt.plot(T_exact,c_exact)
plt.xlabel('T')
plt.ylabel('c')
plt.title('exact solution for c')
plt.legend(loc='best')
plt.show()
'''

plt.plot(T_forS,S/l**2,label='simulation')
plt.title('entropy per spin')
plt.plot(T_forS_exact,S_exact,label='exact solution')
plt.xlabel('T')
plt.ylabel('s')
plt.legend(loc='best')
plt.show()

'''
plt.plot(T_forS_exact,S_exact,label='exact solution')
plt.xlabel('T')
plt.ylabel('s')
plt.title('exact solution for s')
plt.legend(loc='best')
plt.show()
'''

plt.plot(T,np.abs(tau_ms))
plt.title('correlation time (m)')
plt.xlabel('T')
plt.ylabel('$\\tau_m$')
plt.show()

print('time:',t2-t1)

