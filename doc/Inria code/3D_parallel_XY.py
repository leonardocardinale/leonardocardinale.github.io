import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.integrate import cumtrapz


N = 20
J = 1
h = 0
#theta = 0

def calculate_energy(lattice):
    J_interaction_depth = - J * np.sum(np.cos(lattice - np.roll(lattice,shift=1,axis=2)))
    J_interaction_rows = - J * np.sum(np.cos(lattice - np.roll(lattice,shift=1,axis=0)))
    J_interaction_columns = - J * np.sum(np.cos(lattice - np.roll(lattice,shift=1,axis=1)))
    h_coupling = - h * np.sum(np.cos(lattice))
    return h_coupling + J_interaction_depth + J_interaction_rows + J_interaction_columns

def calculate_energy(lattice):
    J_interaction_depth = J/2 * np.sum((lattice - np.roll(lattice,shift=1,axis=2))**2)
    J_interaction_rows = J/2 * np.sum((lattice - np.roll(lattice,shift=1,axis=0))**2)
    J_interaction_columns = J/2 * np.sum((lattice - np.roll(lattice,shift=1,axis=1))**2)
    h_coupling = h/2 * np.sum((lattice)**2)
    return h_coupling + J_interaction_depth + J_interaction_rows + J_interaction_columns

def metropolis(beta,n_iterations=10000):
    
    # Initialize lattice with random spins
    lattice = np.random.uniform(0,2*np.pi,size=(N,N,N))

    # Apply periodic boundary conditions
    lattice[0,:,:] = lattice[-1,:,:]
    lattice[:,0,:] = lattice[:,-1,:]     
    lattice[:,:,0] = lattice[:,:,-1]   

    energy_before = calculate_energy(lattice)
    energia = []
    magx = []
    magy = []
    thetas = []
    norms = []
        
    for iteration in range(n_iterations):
    
        # Choose a random lattice site
        i = np.random.randint(N)
        j = np.random.randint(N)
        k = np.random.randint(N)

        # Generate a random spin
        s_rand = np.random.uniform(0,2*np.pi)
        
        new_lattice = lattice.copy()

        # Update the spin at the chosen site
        new_lattice[i,j,k] = s_rand
        
         # Apply periodic boundary conditions
        lattice[0,:,:] = lattice[-1,:,:]
        lattice[:,0,:] = lattice[:,-1,:]     
        lattice[:,:,0] = lattice[:,:,-1] 

        # Calculate the action after the update
        energy_after = calculate_energy(new_lattice)

        # Decide whether to accept or reject the update
        if np.random.random() < np.exp(beta*(energy_before - energy_after)):
            # Accept the update
            lattice = new_lattice
            energy_before = energy_after
            
        if iteration >= 7*n_iterations/10:
            energia.append(energy_before)
            magx.append(np.cos(lattice).mean())
            magy.append(np.sin(lattice).mean())
            norms.append(np.sqrt(np.cos(lattice).mean()**2 + np.sin(lattice).mean()**2))
            thetas.append(lattice.mean())
                 
    energia =  np.array(energia).mean()
    magx = np.array(magx).mean()
    magy = np.array(magy).mean()
    theta = np.array(thetas).mean()
    #np.sqrt(magx**2+magy**2)
    norm = np.array(norms).mean()

    return [energia, magx, magy,norm,theta]

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    betas = np.linspace(0,0.5,1000)

    with Parallel(n_jobs=num_cores) as parallel:
        #energies_list = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))
        ls = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))

    energies = np.array(ls)[:,0]
    magx = np.array(ls)[:,1]
    magy = np.array(ls)[:,2]
    norm = np.array(ls)[:,3]
    thetas = np.array(ls)[:,4]

    integral = cumtrapz(energies, betas, initial=0)

    free_energies = integral / betas

    specific_heat = - betas[1::]**2 * np.diff(energies)/np.diff(betas)
 
    plt.plot(betas,thetas)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\theta$')
    plt.legend()
    plt.show()

    plt.plot(betas,energies,label='MCMC')

    plt.xlabel(r'$\beta$')
    plt.ylabel('Energy')

    plt.legend()
    plt.show()

    plt.plot(betas,magx,label='x')
    plt.plot(betas,magy,label='y')

    plt.xlabel(r'$\beta$')
    plt.ylabel('Magnetization')

    plt.legend()
    plt.show()

    plt.plot(betas,norm)

    plt.xlabel(r'$\beta$')
    plt.ylabel('Magnetization norm')

    plt.legend()
    plt.show()

    plt.plot(betas,free_energies)

    plt.xlabel(r'$\beta$')
    plt.ylabel('Free energy')

    plt.legend()
    plt.show()

    plt.plot(betas[1::],specific_heat)

    plt.xlabel(r'$\beta$')
    plt.ylabel('Specific heat')

    plt.legend()
    plt.show()
