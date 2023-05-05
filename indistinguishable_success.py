# -*- coding: utf-8 -*-
"""
Created on Mon May  1 10:15:59 2023

@author: jackc
"""

import numpy as np 
import matplotlib.pyplot as plt 
import random

#assigning constants 
ntot = 100
#ntot is max(n1^2+n2^2) where n1 and n2 are wuantum numbers of particle 1 and 2 
#since this is the harmnic oscc basis n1,n2=1,2,...ntot
#initial and final frequencies of the harmonic trap 
omegai =1
omegaf = 3
#introduce the compression ratio 
kappa = omegai/omegaf
#bath temperatures 
betac = 10/omegai
betah = 1/omegai

#this function creates an N by N matrix of random tuple pairs each with a g_i and g_f value
def initialise_interaction_matrix(n_rows, n_cols):
    interaction_matrix = np.empty((n_rows, n_cols), dtype=object)
    
    # Fill the matrix with random integer tuples
    for i in range(n_rows):
        for j in range(n_cols):
            interaction_matrix[i,j] = (random.uniform(0,10), random.uniform(0,10))
    return interaction_matrix

def initialise_p_matrix():
    #the p matrix is a list containing the quantum numbers for each state, 
    #sorted by ntot from lowest to highest 
    #we have just guessed an output size
    p = np.zeros((ntot, 3))
    k1 = 0
    
    for i in range(1, ntot+1):
        for j in range(1, i+1):
            left = i - j**2
            if left <= 0:
                break
            elif np.sqrt(left) % 1 != 0:
                pass
            elif np.sqrt(left) % 1 == 0:
                p[k1, 1:3] = [j, np.sqrt(left)]
                k1 += 1
            if j+1 >= np.sqrt(left):
                break
                
                
            
    p[:, 0] = p[:, 1]**2 + p[:, 2]**2
    p = p[p[:, 0] != 0]
    b = p.shape[0]*p.shape[1]
    P = np.reshape(p, (b//3, 3)) 
    N = int(b/3)
    Hi = np.zeros((N, N))
    Hf = np.zeros((N,N))
    return (Hi, Hf, P, N)

#function to extract eigenvalues
def eigvals_bosons(ntot,betac,betah,omegai,omegaf):
    
    Hi, Hf, P, N = initialise_p_matrix()
    g0 = initialise_interaction_matrix(N, N)
    
    E_i = []
    E_f = []
    occ_vals_i = []
    occ_vals_f = []
    for i in range(len(g0)):
        for j in range(len(g0[0])):
            #empty matrix to store the values in the hamiltonian has now been created 
            #now we will generate the Hamiltonian 
            # Generating Hamiltonian
            for n in range(N):
                for m in range(n, N):
                    #assigning values to particles 1 and 2
                    n1 = P[n, 1]
                    n2 = P[n, 2]
                    m1 = P[m, 1]
                    m2 = P[m, 2]
                    
                    
                    #using if statements to create the f_n and f_m functions
                    #first f_n1
                    if n1 % 2 == 1: #(i.e if n1 is odd)
                        f_n1 = 0
                    else:  #(n1 must be even)
                        f_n1 = ((-1**(n1/2))/(np.math.factorial(n1/2)))*np.sqrt(np.math.factorial(n1)/(2**n1))
                        
                    #now f_n2
                    if n2 %2 == 1: 
                        f_n2 = 0
                    else: 
                        f_n2 = ((-1**(n2/2))/(np.math.factorial(n2/2)))*np.sqrt(np.math.factorial(n2)/(2**n2))
                        
                    #f_m1
                    if m1 %2 == 1:
                        f_m1 = 0
                    else: 
                        f_m1 = ((-1**(m1/2))/(np.math.factorial(m1/2)))*np.sqrt(np.math.factorial(m1)/(2**m1))
                        
                    #f_m2
                    if m2 %2 == 1: #(i.e if n is odd)
                        f_m2 = 0
                    else: 
                        f_m2 = ((-1**(m2/2))/(np.math.factorial(m2/2)))*np.sqrt(np.math.factorial(m2)/(2**m2))
                        
                    # Using if statements to mimic Kronecker deltas
                    delta1 = 1 - np.any(n1 - m1)  #delta_{n1,m1}
                    delta2 = 1 - np.any(n2 - m2)  #delta_{n2,m2}
                    delta3 = 1 - np.any(n1 - m2)  #delta_{n1,m2}
                    delta4 = 1 - np.any(n2 - m1)  #delta_{n2,m1}

                    
                   #generating the Hamiltonian 
                   
                   #initial Hamiltonian
                    if n1 != n2 and m1 != m2:   
                        Hi[n,m] = (m1+m2+1)*(delta1*delta2+delta3*delta4)+(((g0[i,j][0]/(np.sqrt(2*np.pi)))*f_n1*f_n2*f_m1*f_m2))
                    elif n1 == n2 and m1 == m2:     
                        Hi[n,m] = (m1+m2+1)*(delta1)+(((g0[i,j][0]/(np.sqrt(2*np.pi)))*f_n1*f_n2*f_m1*f_m2))
                    else:
                        Hi[n,m] = (((g0[i,j][0]/(np.sqrt(2*np.pi)))*f_n1*f_n2*f_m1*f_m2))
                        
                    Hi[m,n]=Hi[n,m];
                    
                    #final Hamiltonian
                    if n1 != n2 and m1 != m2:   
                        Hf[n,m] = (m1+m2+1)*(delta1*delta2+delta3*delta4)+(((g0[i,j][1]/(np.sqrt(2*np.pi)))*f_n1*f_n2*f_m1*f_m2))
                    elif n1 == n2 and m1 == m2:     
                        Hf[n,m] = (m1+m2+1)*(delta1)+(((g0[i,j][1]/(np.sqrt(2*np.pi)))*f_n1*f_n2*f_m1*f_m2))
                    else:
                        Hf[n,m] = (((g0[i,j][1]/(np.sqrt(2*np.pi)))*f_n1*f_n2*f_m1*f_m2))
                        
                    Hf[m,n]=Hf[n,m];
               
                
            #extract energy eigenvalues 
            E_i.append(sum(np.real(np.linalg.eigvals(Hi)))*omegai)  #have taken np.sort out to see if this helps 
            E_f.append(sum(np.real(np.linalg.eigvals(Hf)))*omegaf)
            
            
            #occupation population eigenvalues
            operatori = np.exp(-betac*Hi)/np.trace(np.exp(-betac*Hi))
            occ_vals_i.append(sum(np.real(np.linalg.eigvals(operatori))))
            operatorf =  np.exp(-betah*Hf)/np.trace(np.exp(-betah*Hf))
            occ_vals_f.append(sum(np.real(np.linalg.eigvals(operatorf))))

    return np.array(E_i).reshape(N,N), np.array(E_f).reshape(N, N), np.array(occ_vals_i).reshape(N, N), np.array(occ_vals_f).reshape(N, N), g0

def engine_performance(E_i, E_f, occ_i, occ_f):
    LAMBDA = E_i/E_f
    occupation_difference = occ_f-occ_i
    #standard efficiency equation (!+Qc/Qh)
    topfrac = LAMBDA*E_f*occupation_difference 
    bottomfrac = E_f*occupation_difference 
    efficiency = 1-topfrac/bottomfrac
    rescaled_efficiency = efficiency/(2/3)
    work_output = E_f*(occupation_difference)+E_i*(-1*occupation_difference)
    #otto work output 
    Wc = (omegai/2)*((1/kappa)-1)*(np.tanh(betac*omegai/2))**-1
    We = (omegaf/2)*((1/kappa)-1)*(np.tanh(betah*omegaf/2))**-1
    otto_work = (Wc+We)
    rescaled_work = work_output/otto_work
    #rescale efficiency with standard otto efficiency 
    return rescaled_efficiency, rescaled_work

def extract_vals(n, g):
    out = []
    for i in range(len(interaction_matrix)):
        for j in range(len(interaction_matrix[0])):
            out.append(interaction_matrix[i,j][n])
    return np.array(out)
    
    
E_i, E_f, occ_i, occ_f, interaction_matrix = eigvals_bosons(ntot,betac,betah,omegai,omegaf)
efficiency = engine_performance(E_i, E_f, occ_i, occ_f)[0]
work = engine_performance(E_i, E_f, occ_i, occ_f)[1]

#%% plotting efficiency
y_vals = extract_vals(1, interaction_matrix)
x_vals = extract_vals(0, interaction_matrix)
fig, ax = plt.subplots()
scatter = ax.scatter(x_vals, y_vals, c=efficiency.ravel(), cmap='viridis', s=100)
cbar = plt.colorbar(scatter)
plt.xlabel('$g_{f}$',fontsize =20)
plt.ylabel('$g_{i}$',fontsize =20)
cbar.set_label('$\eta/\eta_{0}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=14)
cbar.ax.tick_params(labelsize=14)
plt.show()

#%% plotting work output 
y_vals = extract_vals(1, interaction_matrix)
x_vals = extract_vals(0, interaction_matrix)
fig, ax = plt.subplots()
scatter = ax.scatter(x_vals, y_vals, c=work.ravel(), cmap='viridis', s=10)
cbar = plt.colorbar(scatter)
plt.xlabel('$g_{f}$',fontsize =20)
plt.ylabel('$g_{i}$',fontsize =20)
cbar.set_label('$W/W_0$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=14)
cbar.ax.tick_params(labelsize=14)
plt.show()
