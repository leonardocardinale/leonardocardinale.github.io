import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

N = 10

def Phi(arr):
    x, y, z, t = arr
    alpha = x + 1j*y
    beta = z + 1j*t
    return np.array([[alpha, -beta.conjugate()], [beta,alpha.conjugate()]])

def diffeo(t1,t2,t3):
    x1 = np.cos(t1)
    x2 = np.sin(t1) * np.cos(t2)
    x3 = np.sin(t1) * np.sin(t2) * np.cos(t3)
    x4 = np.sin(t1) * np.sin(t2) * np.sin(t3)
    arr = np.array([x1,x2,x3,x4])
    return Phi(arr)

def char(n,t1,t2,t3):
    if t1 != 0:
        return np.sin((n+1) * t1)/np.sin(t1)
    else:
        return n+1

def fourier(f,n):
    def integrand(t1,t2,t3):
        return 1/(2*np.pi**2) * f(diffeo(t1,t2,t3)) * char(n,t1,t2,t3) * np.sin(t1)**2 * np.sin(t2)
    return nquad(integrand,[[0,np.pi],[0,np.pi],[0,2*np.pi]])[0]

def f(U,param):
    return np.exp(2*param*(np.real(U.trace())-2))

def Z(param):
    s = 0
    for n in range(4):
        s += (1/(n+1) * fourier(lambda U:f(U,param),n))**(N**2)
    return s

def mean_field(param):
    return - N**2 * np.log(fourier(lambda U:f(U,param),0))

def mean_wilson(param):
    L = []
    for na in range(4,N**2+1):
        I = np.zeros((2,2), dtype=complex)
        for i in range(2):
            for j in range(2):
                def foo_r(t1,t2,t3):
                    return np.real(1/(2*np.pi**2) * f(diffeo(t1,t2,t3),param) * diffeo(t1,t2,t3)[i,j] * np.sin(t1)**2 * np.sin(t2))
                def foo_imag(t1,t2,t3):
                    return np.imag(1/(2*np.pi**2) * f(diffeo(t1,t2,t3),param) * diffeo(t1,t2,t3)[i,j] * np.sin(t1)**2 * np.sin(t2))
                I[i,j] = nquad(foo_r,[[0,np.pi],[0,np.pi],[0,2*np.pi]])[0] + 1j * nquad(foo_imag,[[0,np.pi],[0,np.pi],[0,2*np.pi]])[0]
        L.append(fourier(lambda U:f(U,param),0)**(-na) * np.real(np.trace(np.linalg.matrix_power(I,na))))
    return L

sigma1 = np.array([[0,1],[1,0]])
sigma2 = np.array([[0,-1j],[1j,0]])
sigma3 = np.array([[1,0],[0,-1]])

# Define the action function
def calculate_action(lattice,beta):
 
    # Calculate the plaquette term in the action
    i_shifted = np.roll(lattice, shift=-1, axis=0)
    j_shifted = np.roll(lattice, shift=-1, axis=1)

    plaq = np.matmul(lattice[..., 0, :, :], j_shifted[..., 1, :, :])
    plaq = np.matmul(plaq, np.conj(i_shifted[..., 0, :, :]).transpose(0, 1, 3, 2))
    plaq = np.matmul(plaq, np.conj(lattice[..., 1, :, :]).transpose(0, 1, 3, 2))

    action_plaquette = np.real(2 - np.trace(plaq, axis1=-2, axis2=-1)).sum()

    # Calculate the total action
    return 2 * beta * action_plaquette

def wilson(lattice,a):
    p = np.eye(2)
    q = np.eye(2)
    r = np.eye(2)
    s = np.eye(2)
    for i in range(-a//2,a//2):
        p = np.matmul(p,lattice[i,-a//2,0])
    for j in range(-a//2,a//2):
        q = np.matmul(p,lattice[a//2,j,1])
    for i in range(a//2,-a//2):
        r = np.matmul(r,np.conj(lattice[i-1,a//2,0]))
    for j in range(a//2,-a//2):
        s = np.matmul(r,np.conj(lattice[-a//2,j-1,1]))
    return np.real(np.trace(p.dot(q.dot(r.dot(s)))))

def X(eps):
    r = np.random.uniform(low=-1/2,high=1/2,size=4)
    x0 =  np.sign(r[0]) * np.sqrt(1 - eps**2)
    x = eps * r[1:]/np.linalg.norm(r[1:])
    return x0 * np.eye(2) + 1j * (x[0] * sigma1 + x[1] * sigma2 + x[2] * sigma3)

def metropolis(beta,n_iterations=100000):
    
    # Initialize lattice with random SU(2) matrices
    lattice = np.zeros((N, N, 2, 2, 2), dtype=np.complex128)
    xvals = np.random.uniform(low=-10, high=10, size=(N, N, 2, 4))
    xvec = xvals / np.linalg.norm(xvals, axis=-1, keepdims=True)  # Normalize each vector

    arr = xvec[..., 0, None, None] * np.eye(2) + 1j * (xvec[..., 1, None, None] * sigma1 + xvec[..., 2, None, None] * sigma2 + xvec[..., 3, None, None] * sigma3)
    lattice[..., 0] = arr[..., 0]
    lattice[..., 1] = arr[..., 1]
   
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
        su2_matrix = X(0.5)
        
        new_lattice = lattice.copy()

        # Update the gauge field at the chosen site
        new_lattice[i, j, k] = np.matmul(su2_matrix, new_lattice[i, j, k])
        
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
    #df.to_csv('characters_SU2.csv', sep=',', index=False)

    energy_df = pd.read_csv('characters_SU2.csv')
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
    strong = np.linspace(0,0.25,100)
    plt.plot(strong,400*strong,label='Strong coupling',)
    #for i in range(len(betas)):
        #plt.plot(side,arr[i,1:],label=f'{betas[i]}')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\langle S \rangle$')
    #plt.xlabel('Side of loop')
    #plt.ylabel('Wilson loop')
    plt.legend()
    plt.show()
    '''

    betas_wilson = np.linspace(1,100,5)

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
