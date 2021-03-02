'''Code implementation for Q2 of cw2'''
from numpy import *
import numpy as np
from scipy import *
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import cla_utils
from Q2_functions import form_quasi_tri, form_tri, solve_Ainv, calculate_u0, calculate_w0, build_f, find_timesteps


# Q2 C 
# show that the LU decomposition has no special banded form

A = form_quasi_tri(2, 6)
m,_=A.shape
print('\nquasi-tridiagonal matrix: \n',A)
cla_utils.LU_inplace(A)
print('\nLU_inplace outputs a full bandwidth matrix: \n', A)


# Q2 F

# settings
M = 100
ntimesteps = 100
deltat = 0.01
nstepsplot = 5  # number of steps after which to plot the solution
nstepsprint = 100  # number of steps after which to save the solution
plot_solution = True 
save_solution = False

xd = np.linspace(0,2*pi,M)

# calling the function that computes the approximation to the solution of the wave equation with periodic boundary u0(x)=sin(2*pi*x)
w, u = find_timesteps(deltat, ntimesteps, M, xd, nstepsplot, nstepsprint, plot_solution, save_solution)

b = np.load('solutions.npy')
print('\nQ2F \nthe saved solutions of the last timestep: \n',b)
