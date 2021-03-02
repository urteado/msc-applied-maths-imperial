from numpy import *
import numpy as np
from scipy import *
import scipy as sp


# Q1 tridiagonal algorithm

# first argument is the value of off-diagonals, second entry is value of diagonal

def LU_tridiagonal(offd, diag, b, output_LU=False):
    m = len(b)

    dg = np.zeros(m)
    dg = dg+diag

    a1 = np.zeros(m)  # for upper diagonal entries of A
    a1 = a1+offd
    a2 = np.zeros(m)  # for lower diagonal entries of A
    a2 = a2+offd

    a11 = np.zeros(m)  # for lower diagonal of L
    a22 = np.zeros(m)  # for diagonal of U
    a22[0] = dg[0]  # set first element as the diagonal element d

    z = np.zeros(m)
    z[0] = b[0]

    # modified LU decomposition merged with the forward substitution: solution to Lz = b
    for i in range(1,m):
        a11[i] = a2[i] / a22[i-1]
        a22[i] = dg[i] - a11[i] * a1[i-1]
        z[i] = b[i] - a11[i]*z[i-1]  # no need to divide by diagonal as it is ones in L

    # backward substitution: solution to Ux = z
    x = np.zeros(m)
    x[m-1] = z[m-1]/a22[m-1]
    for i in range(m-2,-1,-1):
        x[i] = (z[i] - a1[i]*x[i+1]) / a22[i]

    if output_LU:
        L = np.zeros((m,m))
        dg = np.ones(m)
        a11 = a11[1:]
        L = np.diag(dg) + np.diag(a11, -1)

        U = np.zeros((m,m))
        d1 = offd
        d_arr = np.ones(m-1)*d1
        U = np.diag(a22) + np.diag(d_arr, 1)
        return x, L, U

    return x


def gen_tri_symm(c, d, n):
    A = np.zeros((n,n), dtype=complex)
    di = np.ones(n)
    u = np.ones(n-1)
    A = np.diag(d*di) + np.diag(c*u, -1) + np.diag(c*u, 1)
    return A
