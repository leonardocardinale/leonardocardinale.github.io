import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from scipy.special import factorial

N = 10

def fourier(f,n):
    return 1/(2*np.pi) * (quad(lambda t : np.real(f(np.exp(1j*t)) * np.exp(1j*n*t)),0,2*np.pi)[0] + 1j * quad(lambda t : np.imag(f(np.exp(1j*t)) * np.exp(1j*n*t)),0,2*np.pi)[0])

def f(u,param):
    return np.exp(2*param*(np.real(u)-1))

def Z(param):
    s = fourier(lambda u:f(u,param),0)**(N**2)
    for n in range(1,4):
        #s += (fourier(lambda u:f(u,param),n))**(N**2) + (fourier(lambda u:f(u,param),-n))**(N**2)
        s += 2 * (fourier(lambda u:f(u,param),n))**(N**2)
    return s

def coeff(r,beta,J=100):
    #start = np.max(np.array([0,np.ceil(-r)]))
    start = 0
    fact = np.exp(-2*beta)
    j_vals = np.arange(start,start+J+1)
    return fact*np.sum(beta**(2*j_vals+r) * 1/(factorial(j_vals) * factorial(j_vals+r)))

coeff = np.vectorize(coeff)

def Z_var(param,R=10):
    #Z_vals = np.sum(coeff(np.arange(1,1+R), param)**(N**2)) + np.sum(coeff(np.arange(-R,0), param)**(N**2))
    Z_vals = 2*np.sum(coeff(np.arange(1,1+R), param)**(N**2))
    return Z_vals + coeff(0,param)**(N**2)

Z_var = np.vectorize(Z_var)

def mean_field(param):
    return - N**2 * np.log(fourier(lambda U:f(U,param),0))

def mean_wilson(param):
    L = []
    for na in range(4,N**2+1):
        L.append((fourier(lambda U:f(U,param),-1)/fourier(lambda U:f(U,param),0))**na)
    return L

# Define the action function
def calculate_action(lattice,beta):
 
    # Calculate the plaquette term in the action
    i_shifted = np.roll(lattice, shift=-1, axis=0)
    j_shifted = np.roll(lattice, shift=-1, axis=1)

    plaq = lattice[..., 0] * j_shifted[..., 1] * np.conj(i_shifted[..., 0]) * np.conj(lattice[..., 1])

    action_plaquette = np.real(1 - plaq).sum()

    # Calculate the total action
    return 2 * beta * action_plaquette

def wilson(lattice,a):
    p = 1
    for i in range(-a//2,a//2):
        p *= lattice[i,-a//2,0] * np.conj(lattice[i-1,a//2,0])
    for j in range(-a//2,a//2):
        p *= lattice[-a//2,j,1] * np.conj(lattice[a//2,j-1,1])
    return np.real(p)

def X(eps):
    return np.exp(1j * np.random.uniform(low=0,high=2*np.pi))

def metropolis(beta,n_iterations=100000):
    
    # Initialize lattice with random U(1) elements
    lattice = np.exp(1j * np.random.uniform(low = 0, high = 2*np.pi, size=(N,N,2)))

    # Apply periodic boundary conditions
    lattice[0, :, :] = lattice[N-1, :, :]
    lattice[:,0, :] = lattice[:, N-1, :]            

    action_before = calculate_action(lattice,beta)
    energia = []
    #wilsons2 = []
    #wilsons4 = []
    #wilsons6 = []
    #wilsons8 = []
    #wilsons10 = []
    #wilsons = [[] for _ in range(N//2)]
        
    for iteration in range(n_iterations):
    
        # Choose a random lattice site and direction
        i = np.random.randint(N)
        j = np.random.randint(N)
        k = np.random.randint(2)

        # Generate a random U(1) element
        u1_elem = X(0.5)
        
        new_lattice = lattice.copy()

        # Update the gauge field at the chosen site
        new_lattice[i, j, k] = u1_elem * new_lattice[i, j, k]
        
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
            
        if iteration >= 7*n_iterations/10:
            energia.append(action_before)
            #wilsons2.append(wilson(lattice,2))
            #wilsons4.append(wilson(lattice,4))
            #wilsons6.append(wilson(lattice,6))
            #wilsons8.append(wilson(lattice,8))
            #wilsons10.append(wilson(lattice,10))
            #for i in range(N//2):
                #wilsons[i].append(wilson(lattice,2*(i+1)))

    #wilson2 = np.array(wilsons2).mean()
    #wilson4 = np.array(wilsons4).mean()
    #wilson6 = np.array(wilsons6).mean()
    #wilson8 = np.array(wilsons8).mean()
    #wilson10 = np.array(wilsons10).mean()
    #for i in range(N//2):
        #wilsons[i] = np.array(wilsons[i]).mean()     
    energia =  np.array(energia).mean()
    #ls = [energia]
    #for i in range(N//2):
        #ls.append(wilsons[i])
    return energia
    #return [energia, wilson2, wilson4, wilson6, wilson8, wilson10]
    #return ls

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    betas = np.linspace(0, 2, 100)

    '''
    with Parallel(n_jobs=num_cores) as parallel:
        #Z_list = parallel(delayed(Z)(beta) for beta in tqdm(betas))
        Zlog = parallel(delayed(mean_field)(beta) for beta in tqdm(betas))
    #Z_list1 = list(Z_list)
    #Z_arr = np.array(Z_list1)
    mean_arr = np.array(Zlog)
    #energy = - betas[1::] * np.diff(np.log(Z_arr))/np.diff(betas)
    energy_mean = betas[1::] * np.diff(mean_arr)/np.diff(betas)
    #df = pd.DataFrame(data=energy)
    #df.to_csv('characters_U1.csv', sep=',', index=False)

    energy_df = pd.read_csv('characters_U1.csv')
    energy = np.array(energy_df,dtype=complex)
    plt.plot(betas[1::],np.real(energy),label='Characters')
    #plt.plot(betas[1::],-betas[1::]*np.diff(np.log(Z_var(betas)))/np.diff(betas),label='Explicit Fourier')
    plt.plot(betas[1::],energy_mean,label='Mean Field')

    #alt_energy = -betas[1::] * np.diff(np.log(Z_var(betas)))/np.diff(betas)

    #plt.plot(betas[1::],alt_energy, label='closed-form Fourier')

    #betas = np.linspace(0,10,10)

    betas_metro = np.linspace(0,2,10)

    with Parallel(n_jobs=num_cores) as parallel:
        energies_list = parallel(delayed(metropolis)(beta) for beta in tqdm(betas_metro))
        #ls = parallel(delayed(metropolis)(beta) for beta in tqdm(betas))

    #print('finished')

    energies = np.array(energies_list)
    #arr = np.array(ls)
    #side = [2*i for i in range(1,N//2+1)]
    plt.scatter(betas_metro,energies,label='MCMC',marker='+')
    strong = np.linspace(0,0.25,100)
    plt.plot(strong,200*strong,label='Strong coupling')
    #for i in range(len(betas)):
        #plt.plot(side,arr[i,1:],label=f'{betas[i]}')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\langle S \rangle$')
    #plt.xlabel('Side of loop')
    #plt.ylabel('Wilson loop')
    plt.legend()
    plt.show()
    '''

    betas_wilson = np.linspace(0.001,2,5)

    with Parallel(n_jobs=num_cores) as parallel:
        wilson_list = parallel(delayed(mean_wilson)(beta) for beta in tqdm(betas_wilson))
    
    for i in range(len(betas_wilson)):
        arr = - np.log(np.real(np.array(wilson_list[i],dtype=complex)))
        V = arr/np.sqrt(np.arange(4,N**2+1))
        plt.plot(np.sqrt(np.arange(4,N**2+1)),V,label=r'$\beta = $' + f'{betas_wilson[i]}')
    
    plt.xlabel(r'$r$')
    plt.ylabel(r'$V(r)$')
    plt.legend()
    plt.show()
