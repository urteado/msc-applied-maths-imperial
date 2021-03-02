'''function file for Q3'''
from numpy import *
import numpy as np
from scipy import *
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import cla_utils


# Q3 C

# builds R in-place and householder reflectors
def qr_factor_tri(A, kmax=None):
    m,_ = A.shape
    V = np.zeros((2,m-1))
    if kmax is None:
        kmax = m
    for k in range(m-1):
        e1 = np.zeros(2)
        e1[0] = 1
        x = A[k:k+2,k]  # get column slice from diagonal to the bottom
        if x[0] == 0:
            sign = 1  # change default sign x[0] = 0 to x[0] = 1
        else:
            sign = np.sign(x[0])
        # generate householder reflections
        v = sign * np.linalg.norm(x) * e1 + x
        v = v / np.linalg.norm(v)
        A[k:k+2,k:k+3] -= 2*np.dot(np.outer(v,v.transpose()), A[k:k+2,k:k+3])
        V[:,k] = v
    R = 1.0*A  # copy to R
    return R, V


# Wilkinson shift
def shift(T):
    m,_ = T.shape
    a = T[m-1,m-1]
    b = T[m-1,m-2]
    delta = ( T[m-2,m-2] - a )/2
    denom = np.abs(delta) + np.sqrt( np.power(delta,2) + np.power(b,2) )
    if delta == 0:
            sign = 1  # change default sign x[0] = 0 to x[0] = 1
    else:
        sign = np.sign(delta)
    mu = a - sign*np.power(b,2)/denom
    return mu


# unshifted QR algorithm for tridiagonal matrices
def qr_alg_tri(A, maxit, do_shift=False):
    m,_ = A.shape
    Tmm1_array = []
    T_mm1 = np.abs(A[m-1,m-2])
    Tmm1_array.append(T_mm1)
    for i in range(maxit):
        if do_shift:
            mu = shift(A)
            A = A-mu*np.eye(m)
        A, V = qr_factor_tri(A)
        A = A.T  # transpose the matrix to perform the RQ step of the QR algorithm
        for k in range(m-1):            
            A[k:k+2,k:k+3] -= 2 * np.dot(np.outer(V[:,k].transpose(), V[:,k].conj()), A[k:k+2,k:k+3])
            A[k:k+3,k] = A[k,k:k+3].T  # symmetry
        A = A.T  # transpose it back
        
        if do_shift:
            A += mu*np.eye(m)

        T_mm = A[m-1,m-1]
        T_mm1 = np.abs(A[m-1,m-2])
        Tmm1_array.append(T_mm1)
        if T_mm1 < 10e-12:
            return A, T_mm, Tmm1_array
    Tmm1_array = np.array(Tmm1_array)
    return A, T_mm, Tmm1_array


# Q3 E

def script3e(A, maxit, do_shift):
    m, _ = A.shape
    cla_utils.hessenberg(A)  # reduction to tridiagonal
    concat = []
    A, T_mm, Tmm1_array = qr_alg_tri(A, maxit, do_shift) # call it for m
    concat.append(Tmm1_array)
    for i in range(m-1,1,-1):
        A = A[:i,:i]
        A, _, Tmm1_array = qr_alg_tri(A, maxit, do_shift)
        concat.append(Tmm1_array)
        # print('step %s'%str(i),A)
    return A, concat


def script3e_dots(A,maxit, do_shift):
    m, _ = A.shape
    cla_utils.hessenberg(A)  # reduction to tridiagonal
    # call it for m
    A, _, Tmm1_array = qr_alg_tri(A, maxit, do_shift)
    for i in range(m-1,1,-1):
        A = A[:i,:i]
        A, _, Tmm1_array_new = qr_alg_tri(A, maxit, do_shift)
        Tmm1_array = Tmm1_array + Tmm1_array_new
    return Tmm1_array


# Comparison with pure QR
def pure_QR(A, maxit):
    m,_ = A.shape
    Tmm1_array = []
    T_mm1 = np.abs(A[m-1,m-2])
    Tmm1_array.append(T_mm1)
    for i in range(maxit):
        Q, R = np.linalg.qr(A)
        A = R.dot(Q)
        T_mm = A[m-1,m-1]
        T_mm1 = np.abs(A[m-1,m-2])
        Tmm1_array.append(T_mm1)
        if T_mm1 < 10e-12:
            return A, T_mm, Tmm1_array
    Tmm1_array = np.array(Tmm1_array)
    return A, T_mm, Tmm1_array


def script3e_dots_pure(A,maxit):
    m, _ = A.shape
    cla_utils.hessenberg(A)  # reduction to tridiagonal
    # call it for m
    A, _, Tmm1_array = pure_QR(A, maxit)
    for i in range(m-1,1,-1):
        A = A[:i,:i]
        A, _, Tmm1_array_new = pure_QR(A, maxit)
        Tmm1_array = Tmm1_array + Tmm1_array_new
    return Tmm1_array


# implement symmetric matrix A
def implement_A(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = 1/(i+j+3)
    return A


# implement integer symmetric matrix
def implement_B(n):
    B = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            B[i,j] = i+j+5
    return B


def gen_tri_symm(c, d, n):
    A = np.zeros((n,n), dtype=complex)
    di = np.ones(n)
    u = np.ones(n-1)
    A = np.diag(d*di) + np.diag(c*u, -1) + np.diag(c*u, 1)
    return A
