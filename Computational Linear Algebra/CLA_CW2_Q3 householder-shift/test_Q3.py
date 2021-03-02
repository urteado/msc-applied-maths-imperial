'''tests for Q3'''
import pytest
from numpy import *
import numpy as np
from Q3_functions import qr_factor_tri, implement_A, qr_alg_tri, script3e, gen_tri_symm, pure_QR
import cla_utils


@pytest.mark.parametrize('n', [6,17,44])
def test_qr_factor_tri(n):
    random.seed(1302*n)

    # random symmetric matrix + reduction to hessenberg
    A = np.random.randn(n,n)
    A = (A + A.conj().T)/2
    A0 = 1.0*A
    cla_utils.hessenberg(A0)
    A2, _ = qr_factor_tri(A0)

    assert(np.linalg.norm(A2[np.tril_indices(n, -1)])/n**2 < 1.0e-6)  # check for upper triangular
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)  # check for conservation of trace

    # purely tridiagonal matrix, no hessenberg reduction
    c = np.random.randint(1,100)
    d = np.random.randint(1,100)
    B = gen_tri_symm(c, d, n)
    B0 = 1.0*B
    B2, _ = qr_factor_tri(B0)

    assert(np.linalg.norm(B2[np.tril_indices(n, -1)])/n**2 < 1.0e-6)  # check for upper triangular
    assert(np.abs(np.trace(B0) - np.trace(B2)) < 1.0e-6)  # check for conservation of trace


@pytest.mark.parametrize('n', [5,14,50])
def test_qr_alg_tri(n):
    random.seed(1302*n)
    A = np.random.randn(n,n)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    cla_utils.hessenberg(A0)
    tr = np.trace(A0)

    A2, T_mm, _ = qr_alg_tri(A0, 100, do_shift=False)  # passes for both use_shift=True and use_shift=False

    assert(np.linalg.norm(A2[n-1,:n-1]) < 1.0e-6)  # check that the entries of the non-diagonal entries of the last row are below a tolerance
    assert(np.linalg.norm(A2[:n-1,n-1]) < 1.0e-6)  # do the same for the last column
    assert(np.linalg.norm(A2 - A2.conj().T) < 1.0e-6) # check it is still Hermitian
    assert(np.linalg.norm(A2[np.tril_indices(n, -2)])/n**2 < 1.0e-6)  # check for second lower diagonal zeros
    assert(np.abs(tr - np.trace(A2)) < 1.0e-6)  # conservation of trace (sum of eigenvalues - check if eigenvalues unchanged)

    e,_=np.linalg.eig(A0)
    e = np.sort(e)
    eleast = e[n-1]
    assert(T_mm - eleast < 1.0e-12)  # T_mm is the least eigenvalue of the matrix A


@pytest.mark.parametrize('n', [4,16,50])
def test_script3e(n):
    random.seed(1302*n)
    A = np.random.randn(n,n)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    A2, _ = script3e(A0, 100, do_shift=False)
    m,_=A2.shape
    # check that its off-diagonal entries are zero
    assert(np.linalg.norm(A2[m-1,m-2])< 1.0e-10)  
    assert(np.linalg.norm(A2[m-2,m-1])< 1.0e-10)
    # check that it outputs a 2x2 matrix 
    assert(A2.shape == (2, 2))


@pytest.mark.parametrize('n', [4,16,50])
def test_qr_alg_tri_shift(n):
    random.seed(1302*n)
    A = np.random.randn(n,n)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    cla_utils.hessenberg(A0)
    A2, T_mm, _ = qr_alg_tri(A0, 100, do_shift=True)  
    assert(np.linalg.norm(A2[n-1,:n-1]) < 1.0e-6)  # check that the errors of the lower-triangular entries of the last row are below a tolerance
    assert(np.linalg.norm(A2[:n-1,n-1]) < 1.0e-6)  # do the same for columns
    assert(np.linalg.norm(A2 - A2.conj().T) < 1.0e-6)  # check it is still Hermitian
    assert(np.linalg.norm(A2[np.tril_indices(n, -2)])/n**2 < 1.0e-6)  # check for second lower diagonal zeros
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)  # check for conservation of trace (sum of eigenvalues - check if eigenvalues unchanged)

    # test that the T_mm is the least eigenvalue of the matrix A
    e,_=np.linalg.eig(A0)
    e = np.sort(e)
    eleast = e[n-1]
    assert(T_mm - eleast < 1.0e-12)


@pytest.mark.parametrize('n', [4,16,50])
def test_script3e_shift(n):
    random.seed(1382*n)
    A = np.random.randn(n,n)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    A2, _ = script3e(A0,100,do_shift=True)  # outputs a 2x2 matrix A2
    m,_=A2.shape
    # check that its off-diagonal entries are zero
    assert(np.linalg.norm(A2[m-1,m-2])< 1.0e-10)  
    assert(np.linalg.norm(A2[m-2,m-1])< 1.0e-10)
    # check that it outputs a 2x2 matrix 
    assert(A2.shape == (2, 2))

@pytest.mark.parametrize('n', [4,16,50])
def test_pure_QR(n):
    random.seed(1302*n)
    A = random.randn(n, n)
    A = 0.5*(A + A.T)
    A0 = 1.0*A
    cla_utils.hessenberg(A0)
    tr = np.trace(A0)
    A2, _, _ = pure_QR(A0,100)
    assert(np.linalg.norm(A2 - A2.T) < 1.0e-4)  # check it is still Hermitian
    assert(np.linalg.norm(A2[np.tril_indices(n, -2)])/n**2 < 1.0e-5)  # check for second lower diagonal zeros
    assert(np.abs(tr - np.trace(A2)) < 1.0e-6)  # conservation of trace
