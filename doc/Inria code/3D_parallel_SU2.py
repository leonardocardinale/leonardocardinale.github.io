import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.stats import unitary_group
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

sigma1 = np.array([[0,1],[1,0]])
sigma2 = np.array([[0,-1j],[1j,0]])
sigma3 = np.array([[1,0],[0,-1]])

N=10

def increment(coord, mu):
    delta = np.zeros((3,))
    delta[np.abs(mu)] = 1
    return np.mod(np.array(coord) + delta, N).astype(int)

def increment(coord,mu):
    delta = np.zeros(3)
    delta[abs(mu)] = 1
    return ((np.array(coord) + delta) % N).astype(int)

def calculate_action(lattice, beta):
    # Extracting the dimensions of the lattice
    dimensions = lattice.shape[:-2]
    num_dimensions = len(dimensions)

    # Shifted link variables in different directions
    link_shifted = [np.roll(lattice, shift=1, axis=i) for i in range(num_dimensions)]
    link_shifted_conj = [np.roll(lattice, shift=-1, axis=i) for i in range(num_dimensions)]

    # Computing the product of link variables
    prod = lattice
    for i in range(num_dimensions):
        prod = np.matmul(prod, np.matmul(link_shifted[i], np.matmul(np.conjugate(link_shifted[i]), np.conjugate(link_shifted_conj[i]))))

    # Calculating the trace of the product
    trace = np.trace(prod, axis1=-2, axis2=-1)

    # Calculating the sum of the traces and taking the real part
    sum_of_traces = np.real(np.sum(trace, axis=(-2, -1)))

    # Calculating the lattice pure gauge action
    action = 2.0 * beta * (2*3*(3-1)/2 * 10**3 - np.sum(sum_of_traces))

    return action

def X(eps):
    r = np.random.uniform(low=-1/2,high=1/2,size=4)
    x0 =  np.sign(r[0]) * np.sqrt(1 - eps**2)
    x = eps * r[1:]/np.linalg.norm(r[1:])
    return x0 * np.eye(2) + 1j * (x[0] * sigma1 + x[1] * sigma2 + x[2] * sigma3)

def metropolis(beta,n_iterations=100000):

    # Initialize lattice with random SU(2) matrices
    x = unitary_group.rvs(2, size=3*N**3)
    deter = np.linalg.det(x)
    arr = x/(deter[:,np.newaxis,np.newaxis]**(1/2))
    lattice = np.zeros((N,N,N,3,2,2),dtype=complex)
    lattice = arr.reshape((N,N,N,3,2,2))

    # Apply periodic boundary conditions
    lattice[0, :, :, :, :] = lattice[N-1, :, :, :, :]
    lattice[:, 0, :, :, :] = lattice[:, N-1, :, :, :]  
    lattice[:, :, 0, :, :] = lattice[:, :, N-1, :, :]           

    action_before = calculate_action(lattice,beta)
    energia = []
        
    for iteration in range(n_iterations):
    
        # Choose a random lattice site and direction
        i = np.random.randint(N)
        j = np.random.randint(N) 
        k = np.random.randint(N)
        l = np.random.randint(3)

        # Generate a random SU(2) matrix
        su2_matrix = X(0.5)
        
        new_lattice = lattice.copy()

        # Update the gauge field at the chosen site
        new_lattice[i, j, k, l] = np.matmul(su2_matrix, new_lattice[i, j, k, l])
        
        # Apply periodic boundary conditions
        lattice[0, :, :, :, :] = lattice[N-1, :, :, :, :]
        lattice[:, 0, :, :, :] = lattice[:, N-1, :, :, :]  
        lattice[:, :, 0, :, :] = lattice[:, :, N-1, :, :] 

        # Calculate the action after the update
        action_after = calculate_action(new_lattice,beta)

        # Decide whether to accept or reject the update
        if np.random.random() < np.exp(action_before - action_after):
            # Accept the update
            lattice = new_lattice
            action_before = action_after
            
        if iteration >= 9*n_iterations/10:
            energia.append(action_before)
            
    return np.array(energia).mean()

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    betas = np.linspace(0, 100, 10)
    with Parallel(n_jobs=num_cores) as parallel:
        energies_list = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))

    energies = np.array(energies_list)
    plt.plot(betas,energies)
    plt.xlabel(r'$\beta$')
    plt.ylabel('Action')
    plt.show()