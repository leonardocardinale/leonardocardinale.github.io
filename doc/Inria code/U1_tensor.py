import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

N = 10
d = 4

def fourier(f,n,param):
    return 1/(2*np.pi) * (quad(lambda t : np.real(f(np.exp(1j*t)) * np.exp(1j*n*t)),0,2*np.pi)[0] + 1j * quad(lambda t : np.imag(f(np.exp(1j*t)) * np.exp(1j*n*t)),0,2*np.pi)[0])

def f(u,param):
    return np.exp(2*param*(np.real(u)-1))

def Z(param):
    s = fourier(lambda u:f(u,param),0,param)**(N**2)
    for n in range(1,4):
        s += (fourier(lambda u:f(u,param),n,param))**(N**2) + (fourier(lambda u:f(u,param),-n,param))**(N**2)
    return s

def c(param, arr):
    p = 1
    for k in range(len(arr)):
        p *= fourier(lambda u:f(u,param),k,param)
    return p

def tensor(param, arr):
    l = len(arr)
    mu = d + 1 - l
    cst = c(param, arr)**(1/l)
    pos = arr[::2]
    neg = arr[1::2]
    delta = (np.sum(pos) - np.sum(neg) == 0)
    return np.real(cst * delta)

arr = np.random.randint(low=-10,high=10,size=6)
print(tensor(0.4,np.zeros(6)))