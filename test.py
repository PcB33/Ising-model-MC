# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:17:32 2020

@author: Phili
"""

import numpy as np
import matplotlib.pyplot as plt
import random as ran
from scipy.integrate import quad
from scipy.misc import derivative


#exact solution for N->inf per site
J=1
kb=1
T_exact=np.linspace(1.2,4,1000)
beta=1/(kb*T_exact)



L=beta*J#length 15
K=L#length 15
k=1/(np.sinh(2*L)*np.sinh(2*K))#length 15
U_exact=np.empty(T_exact.size)#length 15


def integrand(x,k):
    return 1/np.sqrt(1-4*k*(1+k)**(-2)*np.sin(x)**2)#length 1000

for i in range(T_exact.size):
    integral=quad(integrand,0,np.pi/2,args=k[i])
    U_exact[i]=-J/np.tanh(2*beta[i]*J)*(1+2/np.pi*(2*np.tanh(2*beta[i]*J)**2-1)*integral[0])

#c_exact=np.gradient(U_exact)
c_exact=np.zeros_like(T_exact)
for i in range(T_exact.size-1):
    c_exact[i+1]=(U_exact[i+1]-U_exact[i])/(T_exact[i+1]-T_exact[i])
c_exact[0]=c_exact[1]
Mag_exact=(1-np.sinh(2*beta*J)**(-4))**(1/8)

plt.plot(T_exact,U_exact)
plt.ylabel('U')
plt.show()

plt.plot(T_exact,c_exact)
plt.ylabel('c')
plt.show()

plt.plot(T_exact,Mag_exact)
plt.title('magnetisation per spin')
plt.ylabel('$\mid$m$\mid$')
plt.xlabel('T')
plt.xlim(right=3)
plt.show()

'''
chi_u_log=np.log(chi_u)
N_var_log=np.log(N_var)

alpha,beta=np.polyfit(np.log(N_var)[:90],np.log(chi_u)[:90],1)
print(alpha,beta)

x=np.linspace(0,90,1000)
fit=alpha*x+beta

plt.loglog(N_var[:90],chi_u[:90]/chi_u[0])
plt.plot(x,fit)
plt.show()

plt.plot(N_var_log[:90],chi_u_log[:90]/chi_u_log[0])
plt.show()

'''

rejected=np.empty(0)
print('rejected:',rejected)


x=np.linspace(0,0.1,10000)
y=np.sin(1/x)**4
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('$sin^4(\\frac{1}{x})$')
plt.show()







        

