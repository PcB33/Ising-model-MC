# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random as ran

def approx(N):
    
    points_inside=0
    points_total=0
    
    def point():
        inside=0
        x=ran.random()
        y=ran.random()
        r=np.sqrt(x**2+y**2)
        if (r<=1):
            inside+=1
            
        return np.array([x,y]),inside
    
    points=np.empty((N,2))
    for i in range(N):
        points[i],a=point()
        points_total+=1
        points_inside+=a
    
    pi=4*points_inside/points_total
    
    x=np.linspace(0,1,1000)
    circle=np.sqrt(1-x**2)
    plt.plot(x,circle)
    plt.scatter(points[:,0],points[:,1],marker='.',color='k')
    plt.vlines(np.array([0,1]),ymin=0,ymax=1)
    plt.axis('equal')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()
    
    print('N=',N)
    print('pi=',pi)
    
    return pi

N=np.array([1,10,100,10**3,10**4,10**5,10**6])
x_ax=np.linspace(1,N.size,N.size)
pi_exact=np.pi*np.ones(int(N.size))
pi_num=np.empty(int(N.size))
for i in range(N.size):
    pi_num[i]=approx(N[i])
    
plt.plot(x_ax,pi_num,label='pi_num')
plt.plot(x_ax,pi_exact,label='pi_exact')
plt.legend(loc='best')
plt.show()