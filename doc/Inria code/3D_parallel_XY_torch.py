import matplotlib.pyplot as plt
from numpy.random import uniform
import torch
from torch.distributions import multivariate_normal
import torch.autograd as autograd
import torch.optim as optim
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from math import ceil
import torch.nn as nn
from math import sqrt
import numpy as np
from scipy.integrate import cumtrapz
import torch.multiprocessing as mp

N = 10
J = 1
h = 0
#theta = 0

def U(lattice):
    J_interaction_depth = J/2 * torch.sum((lattice - torch.roll(lattice,shifts=1,dims=2))**2)
    J_interaction_rows = J/2 * torch.sum((lattice - torch.roll(lattice,shifts=1,dims=0))**2)
    J_interaction_columns = J/2 * torch.sum((lattice - torch.roll(lattice,shifts=1,dims=1))**2)
    h_coupling = h/2 * torch.sum((lattice)**2)
    return h_coupling + J_interaction_depth + J_interaction_rows + J_interaction_columns

def U(lattice):
    J_interaction_depth = -J * torch.sum(torch.cos(lattice - torch.roll(lattice,shifts=1,dims=2))**2)
    J_interaction_rows = -J * torch.sum(torch.cos(lattice - torch.roll(lattice,shifts=1,dims=0))**2)
    J_interaction_columns = J * torch.sum(torch.cos(lattice - torch.roll(lattice,shifts=1,dims=1))**2)
    h_coupling = -h * torch.sum(torch.cos(lattice)**2)
    return h_coupling + J_interaction_depth + J_interaction_rows + J_interaction_columns

def metropolis(beta,k_max=2000,time_step=0.1):

    def gradU(configuration):
        return autograd.grad(U(configuration), configuration)[0]

    # Initialize lattice with random spins
    #lattice = torch.rand(size=(N,N,N)) * 2*np.pi

    # Apply periodic boundary conditions
    #lattice[0,:,:] = lattice[-1,:,:]
    #lattice[:,0,:] = lattice[:,-1,:]     
    #lattice[:,:,0] = lattice[:,:,-1]

    normal_distribution_for_langevin = multivariate_normal.MultivariateNormal(loc=torch.zeros(N), covariance_matrix=torch.eye(N))

    array_of_model_configurations = torch.zeros(k_max,N,N,N)
    array_of_model_configurations[0] = torch.rand(size=(N,N,N)) * 2*torch.pi
    array_of_model_configurations.requires_grad = True

    energia = torch.zeros(k_max)
    magx = torch.zeros(k_max)
    magy = torch.zeros(k_max)
    thetas = torch.zeros(k_max)
    norms = torch.zeros(k_max)

    energia[0] = U(array_of_model_configurations[0])
    magx[0] = torch.cos(array_of_model_configurations[0]).mean()
    magy[0] = torch.sin(array_of_model_configurations[0]).mean()
    thetas[0] = array_of_model_configurations[0].mean()
    norms[0] = torch.sqrt(magx[0]**2 + magy[0]**2)

    for k in range(1,k_max):
        
        #i = np.random.randint(N)
        #j = np.random.randint(N)
        #l = np.random.randint(N)

        changes = torch.zeros(k_max,N,N,N)
        changes[k] = array_of_model_configurations[k-1] - time_step * gradU(array_of_model_configurations[k-1]) + torch.sqrt(2*torch.tensor(time_step)) * normal_distribution_for_langevin.sample()
        #changes[k,i,j,l] = torch.rand(1) * 2 * torch.pi 

        energy_before = energia[k-1]
        energy_after = U(changes[k])

        acceptance_rate = torch.exp(beta*(energy_before - energy_after))

        if uniform() < acceptance_rate:
            array_of_model_configurations = array_of_model_configurations + changes

        if k >= 7*k_max/10:
            energia[k] = energy_after
            magx[k] = torch.cos(array_of_model_configurations[k]).mean()
            magy[k] = torch.sin(array_of_model_configurations[k]).mean()
            norms[k] = torch.sqrt(magx[k]**2 + magy[k]**2)
            thetas[k] = array_of_model_configurations[k].mean()
                 
    energia =  energia.detach().mean()
    magx = magx.detach().mean()
    magy = magy.detach().mean()
    theta = thetas.detach().mean()
    norm = norms.detach().mean()
    #torch.sqrt(magx**2+magy**2)
    return [energia, magx, magy,norm,theta]

if __name__ == "__main__":
    mp.set_start_method('spawn')
    num_cores = mp.cpu_count()
    #num_cores = multiprocessing.cpu_count()
    betas = torch.linspace(0,0.1,40)

    #with Parallel(n_jobs=num_cores) as parallel:
        #ls = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))

    with mp.Pool(processes=num_cores) as pool:
        ls = pool.map(metropolis, betas)

    ls = np.array(ls)
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