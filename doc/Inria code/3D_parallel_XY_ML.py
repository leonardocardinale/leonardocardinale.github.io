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
h = 0.001
#theta = 0
d = 3

def U(configuration):
    lattice = configuration.view(N,N,N)
    J_interaction_depth = J/2 * torch.sum((lattice - torch.roll(lattice,shifts=1,dims=2))**2)
    J_interaction_rows = J/2 * torch.sum((lattice - torch.roll(lattice,shifts=1,dims=0))**2)
    J_interaction_columns = J/2 * torch.sum((lattice - torch.roll(lattice,shifts=1,dims=1))**2)
    h_coupling = h/2 * torch.sum((lattice)**2)
    return h_coupling + J_interaction_depth + J_interaction_rows + J_interaction_columns

def U(configuration):
    lattice = configuration.view(N,N,N)
    J_interaction_depth = -J * torch.sum(torch.cos(lattice - torch.roll(lattice,shifts=1,dims=2)))
    J_interaction_rows = -J * torch.sum(torch.cos(lattice - torch.roll(lattice,shifts=1,dims=0)))
    J_interaction_columns = J * torch.sum(torch.cos(lattice - torch.roll(lattice,shifts=1,dims=1)))
    h_coupling = -h * torch.sum(torch.cos(lattice))
    return h_coupling + J_interaction_depth + J_interaction_rows + J_interaction_columns

#def U(configuration):
    #h_coupling = - h*torch.sum(torch.cos(configuration))
    #J_interaction = - J*torch.sum((torch.cos(configuration - torch.roll(configuration, shifts = 1))))
    #return h_coupling + J_interaction

class PlanarFlow(nn.Module):

    def __init__(self, data_dim):
        super().__init__()

        self.u = nn.Parameter(torch.rand(data_dim)/sqrt(data_dim))
        self.w = nn.Parameter(torch.rand(data_dim)/sqrt(data_dim))
        self.b = nn.Parameter(torch.rand(1)/sqrt(data_dim))
        self.h = nn.Tanh()
    
    def h_prime(self, z):
        return 1 - self.h(z) ** 2
    
    def constrained_u(self):
        """
        Constrain the parameters u to ensure invertibility
        """
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))
    
    def forward(self, z):
        u = self.constrained_u()
        hidden_units = torch.matmul(self.w.T, z.T) + self.b

        x = z + u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)

        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)

        log_det = torch.log((1+torch.matmul(u.T, psi)).abs() + 1e-15)

        return x, log_det

    def inverse(self, x):
        u = self.constrained_u()
        hidden_units = (torch.matmul(self.w.T, x.T) + self.b).T
        z = x - u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)
        log_det = -torch.log((1 + torch.matmul(u.T, psi)).abs() + 1e-15)
        return z, log_det

class LayeredPlanarFlow(nn.Module):

    def __init__(self, data_dim, flow_length = 16):
        super().__init__()

        self.layers = nn.Sequential(
            *(PlanarFlow(data_dim) for _ in range(flow_length)))

    def forward(self, z):
        log_det_sum = 0
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det
        return z, log_det_sum
    
    def inverse(self, z):
        log_det_sum = 0
        for layer in self.layers:
            z, log_det = layer.inverse(z)
            log_det_sum += log_det
        return z, log_det_sum

def metropolis(beta,k_max=350,time_step=0.1,k_lang=20,epsilon=1e-2,n=10,avg_size=1000):

    flow = LayeredPlanarFlow(N**d)
    base_distribution =  multivariate_normal.MultivariateNormal(loc=torch.zeros(N**d), covariance_matrix=torch.eye(N**d))

    def gradU(configuration):
        return autograd.grad(U(configuration), configuration)[0]

    normal_distribution_for_langevin = multivariate_normal.MultivariateNormal(loc=torch.zeros(N**d), covariance_matrix=torch.eye(N**d))

    array_of_model_configurations = torch.zeros(k_max,n,N**d)
    array_of_model_configurations[0] = torch.rand(size=(n,N**d)) * 2*torch.pi
    array_of_model_configurations.requires_grad = True

    optimizer = optim.Adam(flow.parameters(), lr=epsilon)

    energia = torch.zeros(k_max)
    magx = torch.zeros(k_max)
    magy = torch.zeros(k_max)
    thetas = torch.zeros(k_max)
    norms = torch.zeros(k_max)

    energia[0] = U(array_of_model_configurations[0,0]).mean()
    magx[0] = torch.cos(array_of_model_configurations[0,0]).mean()
    magy[0] = torch.sin(array_of_model_configurations[0,0]).mean()
    thetas[0] = array_of_model_configurations[0,0].mean()
    norms[0] = torch.sqrt(magx[0]**2 + magy[0]**2)

    for k in range(1,k_max):
        iteration_successful = False
        while not iteration_successful:
            try:
                for i in range(n):
                    if k%k_lang == 0:
                        proposed_change  = flow.forward(base_distribution.sample())[0]
                        if torch.isnan(proposed_change).any():
                            raise ValueError("NaN detected in flow forward output")

                        changes = torch.zeros(k_max,n,N**d)
                        #changes[k,i] = flow(base_distribution.sample())[0]
                        changes[k,i] = proposed_change
                        acceptance_rate = torch.exp(log_rho_hat(changes[k-1,i])
                                            - log_rho_hat(changes[k,i])
                                            + beta*U(array_of_model_configurations[k-1, i])
                                            - beta*U(changes[k, i]))
                        if torch.isnan(acceptance_rate):
                            raise ValueError("NaN detected in acceptance rate calculation")
                    else:
                        changes = torch.zeros(k_max,n,N**d)
                        changes[k,i] = array_of_model_configurations[k-1,i] - time_step * gradU(array_of_model_configurations[k-1,i]) + torch.sqrt(2*torch.tensor(time_step)) * normal_distribution_for_langevin.sample()
                        energy_before = energia[k-1]
                        energy_after = U(changes[k,i])
                        acceptance_rate = torch.exp(beta*(energy_before - energy_after))

                    if uniform() < acceptance_rate:
                        array_of_model_configurations = array_of_model_configurations + changes


                def log_rho_hat(x):
                    # Ensure x is valid for flow.inverse and log_prob
                    x_inv, log_det = flow.inverse(x)
                    return base_distribution.log_prob(x_inv) + log_det
        
                if k >= 7*k_max/10:
                    energia[k] = energy_after
                    magx[k] = torch.cos(array_of_model_configurations[k,0]).mean()
                    magy[k] = torch.sin(array_of_model_configurations[k,0]).mean()
                    norms[k] = torch.sqrt(magx[k]**2 + magy[k]**2)
                    thetas[k] = array_of_model_configurations[k,0].mean()

                # OPTIMISATION
                optimizer.zero_grad()
                x = array_of_model_configurations[k-1, :].clone().detach().requires_grad_(False)

                # Ensure x is valid for flow.inverse
                x_inv, inv_log_det = flow.inverse(x)
                if torch.isnan(x_inv).any() or torch.isnan(inv_log_det).any():
                    raise ValueError("NaN values detected in flow.inverse outputs.")

                loss = - (base_distribution.log_prob(x_inv) + inv_log_det).mean()
                if torch.isnan(loss):
                    raise ValueError("NaN values detected in loss calculation.")

                loss.backward()
                optimizer.step()

            except ValueError as e:
                print(f"NaN detected, retrying iteration {k}: {e}")
        
            iteration_successful = True
                
    energia =  energia.detach().mean()
    magx = magx.detach().mean()
    magy = magy.detach().mean()
    theta = thetas.detach().mean()
    norm = norms.detach().mean()
    #torch.sqrt(magx**2+magy**2)

    # VALIDATION

    mag = torch.zeros(avg_size)
    energ = torch.zeros(avg_size)
    for i in range(avg_size):
        samp = flow(base_distribution.sample())[0][0]
        mag[i] = torch.sqrt(torch.mean(torch.cos(samp))**2 + torch.mean(torch.sin(samp))**2)
        energ[i] = U(samp)
    mag_val = float(mag.mean())
    energ_val = float(energ.mean())

    return [energia, magx, magy,norm,theta,mag_val,energ_val]

    return res

if __name__ == "__main__":
    mp.set_start_method('spawn')
    num_cores = mp.cpu_count()
    #num_cores = multiprocessing.cpu_count()
    betas = torch.linspace(0,0.1,40)

    #with Parallel(n_jobs=num_cores) as parallel:
        #ls = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))

    with mp.Pool(processes=num_cores) as pool:
        ls = list(tqdm(pool.imap(metropolis, betas), total=len(betas)))
        #ls = pool.map(metropolis, betas)

    ls = np.array(ls)
    energies = np.array(ls)[:,0]
    magx = np.array(ls)[:,1]
    magy = np.array(ls)[:,2]
    norm = np.array(ls)[:,3]
    thetas = np.array(ls)[:,4]

    mag_valid = np.array(ls)[:,5]
    energ_valid = np.array(ls)[:,6]

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

    plt.plot(betas,energ_valid)

    plt.xlabel(r'$\beta$')
    plt.ylabel('Energy validation')

    plt.legend()
    plt.show()

    plt.plot(betas,mag_valid)

    plt.xlabel(r'$\beta$')
    plt.ylabel('Magnetization validation')

    plt.legend()
    plt.show()