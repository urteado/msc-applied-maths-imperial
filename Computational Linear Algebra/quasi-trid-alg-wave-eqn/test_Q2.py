'''Tests for Q2 of cw2'''
import pytest
from numpy import random
from numpy import *
import numpy as np
from Q2_functions import form_quasi_tri, solve_A1inv, solve_Ainv, find_timesteps


@pytest.mark.parametrize('n1, n2, n', [(1, 4, 4), (1, 10, 10), (1, 50, 100)])
def test_solve_Ainv(n1,n2,n):
    random.seed(4933*n1+3953*n2+1302*n)
    C = random.randint(n1,n2)

    A = form_quasi_tri(C, n)  # quasi-tridiagonal - tridiagonal with entries in the corners
    AC = 1.0*A 

    # generate the vectors
    v1 = np.zeros(n)
    v1[n-1] = 1
    v2 = np.zeros(n)
    v2[0] = 1
    u1 = np.zeros(n)
    u1[0] = -C
    u2 = np.zeros(n)
    u2[n-1] = -C

    # initialise a random vector b
    b = random.randn(n)

    # get solution with solve_Ainv
    x1 = solve_Ainv(C, b, u1, v1, u2, v2)

    r = np.dot(AC,x1)-b  # get r = Ax-b
    assert(np.linalg.norm(r) < 1.0e-6)

    x2 = np.linalg.solve(AC,b)  # solving using built-in function

    # check solve_Ainv gives the same solution as linalg.solve
    assert(np.linalg.norm(x2-x1) < 1.0e-6)
