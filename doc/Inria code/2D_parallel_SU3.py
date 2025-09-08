import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from scipy.stats import unitary_group

N = 10

def char_aux(p,q,A,B):
    term1 = np.exp(1j*p*A-1j*q*B) - np.exp(-1j*q*A+1j*p*B)
    term2 = np.exp(-1j*p*(A+B)) * (np.exp(-1j*q*A)-np.exp(-1j*q*B))
    term3 = np.exp(1j*q*(A+B)) * (np.exp(1j*p*B)-np.exp(1j*p*A))
    return -1j * (term1 + term2 + term3)

def plaq(beta,A,B):
    vp1, vp2, vp3 = np.exp(1j*A), np.exp(1j*B), np.exp(1j*(-A-B))
    return np.exp(2*beta*np.real((vp1+vp2+vp3-3)))

def s_(A,B):
    return 8*np.sin(0.5*(A-B))*np.sin(0.5*(A+2*B))*np.sin(0.5*(2*A+B))

def coeff(beta,p,q):
    rea = nquad(lambda A,B:np.real(plaq(beta,A,B) * np.conj(char_aux(p,q,A,B)) * s_(A,B)/ (4*np.pi**2)),[[-np.pi,np.pi],[-np.pi,np.pi]])[0]
    imagi = nquad(lambda A,B:np.imag(plaq(beta,A,B) * np.conj(char_aux(p,q,A,B)) * s_(A,B)/ (4*np.pi**2)),[[-np.pi,np.pi],[-np.pi,np.pi]])[0]
    return rea + 1j * imagi

coeff_vec = np.vectorize(coeff)

def Z(beta):
    s = 0
    p = np.arange(1,6)
    q = np.arange(1,6)
    P,Q = np.meshgrid(p,q)
    d = 1/2 * P * Q * (P + Q)
    arr = (1/d * coeff_vec(beta,P,Q))**(N**2)
    return np.real(np.sum(arr))

def mean_field(beta):
    return - N**2 * np.log(coeff(beta,1,1))

# Define the action function
def calculate_action(lattice,beta):
 
    # Calculate the plaquette term in the action
    i_shifted = np.roll(lattice, shift=-1, axis=0)
    j_shifted = np.roll(lattice, shift=-1, axis=1)

    plaq = np.matmul(lattice[..., 0, :, :], j_shifted[..., 1, :, :])
    plaq = np.matmul(plaq, np.conj(i_shifted[..., 0, :, :]).transpose(0, 1, 3, 2))
    plaq = np.matmul(plaq, np.conj(lattice[..., 1, :, :]).transpose(0, 1, 3, 2))

    action_plaquette = np.real(3 - np.trace(plaq, axis1=-2, axis2=-1)).sum()

    # Calculate the total action
    return 2 * beta * action_plaquette

def wilson(lattice,a):
    p = np.eye(3)
    q = np.eye(3)
    r = np.eye(3)
    s = np.eye(3)
    for i in range(-a//2,a//2):
        p = np.matmul(p,lattice[i,-a//2,0])
    for j in range(-a//2,a//2):
        q = np.matmul(p,lattice[a//2,j,1])
    for i in range(a//2,-a//2):
        r = np.matmul(r,np.conj(lattice[i-1,a//2,0]))
    for j in range(a//2,-a//2):
        s = np.matmul(r,np.conj(lattice[-a//2,j-1,1]))
    return np.real(np.trace(p.dot(q.dot(r.dot(s)))))

def Phi(a,b,c,d):
    x = a + 1j * b
    y = c + 1j * d
    return np.array([[x,-np.conj(y)],[y,np.conj(x)]])

def X():
    t1 = np.random.uniform(low=0,high=np.pi,size=3)
    t2 = np.random.uniform(low=0,high=np.pi,size=3)
    t3 = np.random.uniform(low=0,high=2*np.pi,size=3)
    y1 = np.cos(t1)
    y2 = np.sin(t1) * np.cos(t2)
    y3 = np.sin(t1) * np.sin(t2) * np.cos(t3)
    y4 = np.sin(t1) * np.sin(t2) * np.sin(t3)
    arr1 = Phi(y1[0],y2[0],y3[0],y4[0])
    arr2 = Phi(y1[1],y2[1],y3[1],y4[1])
    arr3 = Phi(y1[2],y2[2],y3[2],y4[2])
    R, S, T = np.zeros((3,3),dtype=complex), np.zeros((3,3),dtype=complex), np.zeros((3,3),dtype=complex)
    R[:2,:2] = arr1[:]
    R[2,2] = 1
    S = np.array([[arr2[0,0], 0, arr2[0,1]],[0, 1, 0],[arr2[1,0], 0, arr2[1,1]]])
    T[0,0] =1
    T[1:3,1:3] = arr3[:]
    X = np.zeros((3,3),dtype=complex)
    X = np.matmul(R,np.matmul(S,T))
    return X

def metropolis(beta,n_iterations=100000):
    
    # Initialize lattice with random SU(3) matrices
    x = unitary_group.rvs(3, size=2*N**2)
    deter = np.linalg.det(x)
    arr = x/(deter[:,np.newaxis,np.newaxis]**(1/3))
    lattice = np.zeros((N,N,2,3,3),dtype=complex)
    lattice[:,:] = arr.reshape((N,N,2,3,3))
   
    # Apply periodic boundary conditions
    lattice[0, :, :] = lattice[N-1, :, :]
    lattice[:,0, :] = lattice[:, N-1, :]            

    action_before = calculate_action(lattice,beta)
    energia = []
    wilsons2 = []
    wilsons4 = []
    wilsons6 = []
    wilsons8 = []
    wilsons10 = []
        
    for iteration in range(n_iterations):
    
        # Choose a random lattice site and direction
        i = np.random.randint(N)
        j = np.random.randint(N)
        k = np.random.randint(2)

        # Generate a random SU(2) matrix
        su3_matrix = X()
        
        new_lattice = lattice.copy()

        # Update the gauge field at the chosen site
        new_lattice[i, j, k] = np.matmul(su3_matrix, new_lattice[i, j, k])
        
        # Apply periodic boundary conditions
        new_lattice[0, :, :] = new_lattice[N-1, :, :]
        new_lattice[:,0, :] = new_lattice[:, N-1, :]

        # Calculate the action after the update
        action_after = calculate_action(new_lattice,beta)

        # Decide whether to accept or reject the update
        if np.random.random() < np.exp(action_before - action_after):
            # Accept the update
            lattice = new_lattice
            action_before = action_after
            
        if iteration >= 9*n_iterations/10:
            energia.append(action_before)
            #wilsons2.append(wilson(lattice,2))
            #wilsons4.append(wilson(lattice,4))
            #wilsons6.append(wilson(lattice,6))
            #wilsons8.append(wilson(lattice,8))
            #wilsons10.append(wilson(lattice,10))
            
    #wilson2 = np.array(wilsons2).mean()
    #wilson4 = np.array(wilsons4).mean()
    #wilson6 = np.array(wilsons6).mean()
    #wilson8 = np.array(wilsons8).mean()
    #wilson10 = np.array(wilsons10).mean()        
    energia =  np.array(energia).mean()
    return energia
    #return [energia, wilson2, wilson4, wilson6, wilson8, wilson10]

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    betas = np.linspace(0, 2, 100)
    with Parallel(n_jobs=num_cores) as parallel:
        #Z_list = parallel(delayed(Z)(beta) for beta in tqdm(betas))
        Zlog = parallel(delayed(mean_field)(beta) for beta in tqdm(betas))
    #Z_list1 = list(Z_list)
    #Z_arr = np.array(Z_list1)
    mean_arr = np.array(Zlog)
    #energy = - betas[1::] * np.diff(np.log(Z_arr))/np.diff(betas)
    energy_mean = betas[1::] * np.diff(mean_arr)/np.diff(betas)
    #df = pd.DataFrame(data=energy)
    #df.to_csv('characters_SU3.csv', sep=',', index=False)

    energy_df = pd.read_csv('characters_SU3.csv')
    energy = np.array(energy_df)

    plt.plot(betas[1::],energy,label='Characters')
    plt.plot(betas[1::],energy_mean,label='Mean Field')

    #betas = np.linspace(0,10,4)

    betas_metro = np.linspace(0,2,10)

    with Parallel(n_jobs=num_cores) as parallel:
        energies_list = parallel(delayed(metropolis)(beta) for beta in tqdm(betas_metro))
        #ls = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))

    energies = np.array(energies_list)
    #arr = np.array(ls)
    #side = [2,4,6,8,10]
    plt.scatter(betas_metro,energies,label='MCMC',marker='+')
    strong = np.linspace(0,0.5,100)
    plt.plot(strong,600*strong,label='Strong coupling')
    #for i in range(len(betas)):
        #plt.plot(side,arr[i,1:],label=f'{betas[i]}')
    plt.xlabel(r'$\beta$')
    plt.ylabel('Action')
    #plt.xlabel('Side of loop')
    #plt.ylabel('Wilson loop')
    plt.legend()
    plt.show()


