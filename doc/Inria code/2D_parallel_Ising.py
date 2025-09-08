import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

N = 10

def calculate_energy(lattice):
    i_shiftedp = np.roll(lattice, shift=1, axis=0)
    i_shiftedm = np.roll(lattice, shift=-1, axis=0)
    j_shiftedp = np.roll(lattice, shift=1, axis=1)
    j_shiftedm = np.roll(lattice, shift=-1, axis=1)
    return - np.sum(i_shiftedp * lattice) - np.sum(i_shiftedm * lattice) - np.sum(j_shiftedp * lattice) - np.sum(j_shiftedm * lattice)

def metropolis(beta,n_iterations=1000000):
    
    # Initialize lattice with random spins
    lattice = np.random.choice(np.array([-1,1]),size=(N,N))

    # Apply periodic boundary conditions
    lattice[0,:] = lattice[N-1,:]
    lattice[:,0] = lattice[:,N-1]          

    energy_before = calculate_energy(lattice)
    energia = []
    mag = []
        
    for iteration in range(n_iterations):
    
        # Choose a random lattice site
        i = np.random.randint(N)
        j = np.random.randint(N)

        # Generate a random spin
        s_rand = np.random.choice(np.array([-1,1]))
        
        new_lattice = lattice.copy()

        # Update the spin at the chosen site
        new_lattice[i,j] = s_rand
        
        # Apply periodic boundary conditions
        new_lattice[0,:] = new_lattice[N-1,:]
        new_lattice[:,0] = new_lattice[:,N-1]

        # Calculate the action after the update
        energy_after = calculate_energy(new_lattice)

        # Decide whether to accept or reject the update
        if np.random.random() < np.exp(beta*(energy_before - energy_after)):
            # Accept the update
            lattice = new_lattice
            energy_before = energy_after
            
        if iteration >= 9*n_iterations/10:
            energia.append(energy_before)
            mag.append(lattice[int(N/2)])
                 
    energia =  np.array(energia).mean()
    mag = np.array(mag).mean()
    return [energia, mag]

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    betas = np.linspace(0,2,100)

    with Parallel(n_jobs=num_cores) as parallel:
        #energies_list = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))
        ls = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))

    energies = np.array(ls)[:,0]
    magnetization = np.array(ls)[:,1]
 
    plt.plot(betas,energies,label='MCMC')

    plt.xlabel(r'$\beta$')
    plt.ylabel('Energy')

    plt.legend()
    plt.show()

    plt.plot(betas,magnetization,label='MCMC')

    plt.xlabel(r'$\beta$')
    plt.ylabel('Magnetization')

    plt.legend()
    plt.show()