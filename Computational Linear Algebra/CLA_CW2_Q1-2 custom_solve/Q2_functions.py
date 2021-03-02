''' functions for Q2 '''

from numpy import *
import numpy as np
from scipy import *
import scipy as sp
from Q1 import LU_tridiagonal
import matplotlib
import matplotlib.pyplot as plt
import cla_utils


def form_quasi_tri(C, n):
    A = np.zeros((n,n))
    di = np.ones(n)
    u = np.ones(n-1)
    A = np.diag((1+2*C)*di) + np.diag(-C*u, -1) + np.diag(-C*u, 1)
    A[0,n-1] = -C
    A[n-1,0] = -C
    return A


def form_tri(C,n):
    A = np.zeros((n,n))
    di = np.ones(n)
    u = np.ones(n-1)
    A = np.diag((1+2*C)*di) + np.diag(-C*u, -1) + np.diag(-C*u, 1)
    return A


# Q2 F

def solve_A1inv(C, b, u1, v1):
    offd = -C
    diag = 1+2*C

    # first solve for y in Ty = b, to get y = T-1b
    y = LU_tridiagonal(offd, diag, b)

    # solve for w in Tw=u1, to get w = T-1u1
    w = LU_tridiagonal(offd, diag, u1)

    x = y - w.dot( np.dot(v1.T , y) ) / (1 + np.dot(v1.T , w))

    return x


def solve_Ainv(C, b, u1, v1, u2, v2):
    # first solve for y in A1y = b, to get y = A1^(-1)b
    y2 = solve_A1inv(C, b, u1, v1)

    # solve for w in A1w=u2, to get w = A1^(-1)u2
    w2 = solve_A1inv(C, u2, u1, v1)

    x = y2 - w2.dot( np.dot( v2.T , y2 ) ) / ( 1 + np.dot( v2.T , w2 ) )

    return x


# Q2 G

def calculate_u0(M):
    u0 = np.zeros(M)
    for i in range(M):
        u0[i] = np.sin(2*np.pi*(i+2)/M)
    return u0


def calculate_w0(M):
    w0 = np.zeros(M)
    for i in range(M):
        w0[i] = 2*np.pi*np.cos(2*np.pi*(i+2)/M)
    return w0


def build_f(M, dt, u, w):
    f = np.zeros((M,1))
    dx = 1/M
    for i in range(-1,M-1):
        f[i] = w[i] + (dt/(dx**2)) * (u[i+1] - 2*u[i] + u[i-1]) + ((dt/(2*dx))**2)*(w[i+1] - 2*w[i] + w[i-1]) # using previous timestep
    return f



def find_timesteps(dt, ntimesteps, M, xd, nstepsplot, nstepsprint, plot=False, save=False):
    # initialise vectors for f
    # outfile = TemporaryFile()
    dx = 1/M
    C = np.power(dt/(2*dx),2)

    # set the u1,u2,v1,v2
    u1 = np.zeros(M)
    u1[0] = -C
    u2 = np.zeros(M)
    u2[M-1] = -C

    v1 = np.zeros(M)
    v1[M-1] = 1
    v2 = np.zeros(M)
    v2[0] = 1

    u = calculate_u0(M)
    wold = calculate_w0(M)

    for i in range(ntimesteps):
        f = build_f(M, dt, u, wold)
        wnew = solve_Ainv(C, f, u1, v1, u2, v2)
        u += (dt/2)*(wold + wnew)
        wold = 1.0*wnew
        if plot:
            if i%nstepsplot==0:
                plt.plot(xd,u,label='timestep '+ str(i))
                plt.title('Every ' + str(nstepsplot) + ' timesteps')
        if save:
            if i%nstepsprint==0:
                np.save('solutions.npy', wold)

    if plot:
        plt.legend(loc='upper right')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.show()

    return wold, u
