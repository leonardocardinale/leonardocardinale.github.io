import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

N = 100

def energy(beta):
    return - N * np.tanh(beta)

def calculate_energy(lattice):
    i_shifted = np.roll(lattice, shift=-1, axis=0)
    return - np.sum(i_shifted * lattice)

def metropolis(beta,n_iterations=100000):
    
    # Initialize lattice with random spins
    lattice = np.random.choice(np.array([-1,1]),size=N)

    # Apply periodic boundary conditions
    lattice[0] = lattice[-1]          

    energy_before = calculate_energy(lattice)
    energia = []
        
    for iteration in range(n_iterations):
    
        # Choose a random lattice site
        i = np.random.randint(N)

        # Generate a random spin
        s_rand = np.random.choice(np.array([-1,1]))
        
        new_lattice = lattice.copy()

        # Update the spin at the chosen site
        new_lattice[i] = s_rand
        
        # Apply periodic boundary conditions
        new_lattice[0] = new_lattice[-1]

        # Calculate the action after the update
        energy_after = calculate_energy(new_lattice)

        # Decide whether to accept or reject the update
        if np.random.random() < np.exp(beta*(energy_before - energy_after)):
            # Accept the update
            lattice = new_lattice
            energy_before = energy_after
            
        if iteration >= 9*n_iterations/10:
            energia.append(energy_before)
                 
    energia =  np.array(energia).mean()
    return energia

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    betas = np.linspace(0,2,100)

    plt.plot(betas,energy(betas),label='theory')

    with Parallel(n_jobs=num_cores) as parallel:
        energies_list = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))

    energies = np.array(energies_list)
 
    plt.plot(betas,energies,label='MCMC')

    plt.xlabel(r'$\beta$')
    plt.ylabel('Energy')

    plt.legend()
    plt.show()