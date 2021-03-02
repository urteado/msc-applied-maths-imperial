from numpy import *
import numpy as np
from scipy import *
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import cla_utils
from Q3_functions import implement_A, implement_B, qr_factor_tri, qr_alg_tri, script3e, shift, script3e_dots, script3e_dots_pure


# Q3 D observations about A

m = 5
A = implement_A(m)
e,_=np.linalg.eig(A)
print(np.round(A,decimals=3))
print(A)
cla_utils.hessenberg(A)
print(A)
print('\nhessenberg form\n',np.round(A,decimals=6))

A0, T_mm, Tmm1_array = qr_alg_tri(A,50)
print('\nafter qr_alg_tri\n',A0)
print('\nafter qr_alg_tri\n',np.round(A0,decimals=10))

print('Q3d \nafter applying qr_alg_tri, T_mm = ',T_mm)

e = np.sort(e)
print('\nnumpy eigenvalues, ascending order = ', e)
eleast = e[0]
print('\n|built-in least eig - T_mm| = ', np.abs(T_mm - eleast))


# Pure QR algorithm for tridiagonal matrices with deflation and shift

# shift setting, set to True to do Wilkinson shift
do_shift = True
if do_shift:
    shift = 'with shift'
else:
    shift = 'no shift'  # for plot title


# MATRIX A

n = 5  # we can change the size of the matrices we are investigating
A = implement_A(n)
A3, concata = script3e(A, 1000, do_shift)

# plot convergence of each submatrix separately
plt.yscale('log')
for i in range(len(concata)):
    plt.plot(concata[i], label='submatrix size %s'%str(5-i))
plt.xlabel('position in the array: length is number of iterations') # because it is formed in the while k<maxit loop
plt.ylabel('$|T_{m,m-1}|$')
plt.title('$A_{i,j}=1/(i+j+1)$ '+shift)
plt.legend()
plt.show()  # line plot


A = implement_A(n)
Tmm1_array = script3e_dots(A, 1000, do_shift)

# plot all iterations at once
plt.yscale('log')
plt.plot(Tmm1_array, 'o')
plt.xlabel('iteration')
plt.ylabel('$|T_{m,m-1}$|')
plt.title('$A_{i,j}=1/(i+j+1)$ '+ shift)
plt.show()  # scatter plot


# # Comparison with the pure QR
# A = implement_A(n)
# pure_array = script3e_dots_pure(A, 1000)
# plt.yscale('log')
# plt.plot(pure_array, 'o')
# plt.xlabel('iteration')
# plt.ylabel('$|T_{m,m-1}$|')
# plt.title('pure QR $A_{i,j}=1/(i+j+1)$ (no shift)')
# plt.show()  # scatter plot


# MATRIX B

B = implement_B(n)
B0 = 1.0*B
print('\nQ3E\nB\n',B0)
cla_utils.hessenberg(B0)

print('\nB hessenberg \n', B0)

_, concatb = script3e(B, 1000, do_shift)
plt.yscale('log')
for i in range(len(concatb)):
    plt.plot(concatb[i], label='submatrix size %s'%str(5-i))
plt.xlabel('position in the array: length is number of iterations') # because it is formed in the while k<maxit loop
plt.ylabel('$|T_{m,m-1}|$')
plt.title('$B_{i,j}=i+j+3$ '+shift)
plt.legend()
plt.show()  # line plot

B = implement_B(n)
Tmm1_array = script3e_dots(B, 1000, do_shift)
plt.yscale('log')
plt.plot(Tmm1_array, 'o')
plt.xlabel('iteration')
plt.ylabel('$|T_{m,m-1}$|')
plt.title('$B_{i,j}=i+j+3$ '+shift)
plt.show()  # scatter plot


# MATRIX C

# random symmetric matrix
C = np.random.randint(5, size=(n,n))
C = np.random.randn(5,5)
C = (C + C.T)/2
C0=1.0*C
print('\nC\n',C)

_, concatc = script3e(C0, 1000, do_shift)
plt.yscale('log')
for i in range(len(concatc)):
    plt.plot(concatc[i], label='submatrix size %s'%str(5-i))
plt.xlabel('position in the array: length is number of iterations') # because it is formed in the while k<maxit loop
plt.ylabel('$|T_{m,m-1}|$')
plt.title('random symmetric matrix ' + shift)
plt.legend()
plt.show()  # line plot

C1=1.0*C
Tmm1_array = script3e_dots(C1, 1000, do_shift)
plt.yscale('log')
plt.plot(Tmm1_array, 'o')
plt.xlabel('iteration')
plt.ylabel('$|T_{m,m-1}$|')
plt.title('random symmetric matrix ' + shift)
plt.show()  # scatter plot



# Q3 G

# initialise matrix A
d = np.arange(14,-1,-1)
D = np.diag(d)
O = np.ones((15,15))
A = D + O

# investigate it
A0=1.0*A
cla_utils.hessenberg(A0)
print('\nQ3g \nhessenberg of A = D+O: \n',A0)
diag = np.diag(A0)
print('\ndiagonal elements of hessenberg(A): \n', diag)
m,_=A0.shape


A1=1.0*A
A10, concata = script3e(A1,100, do_shift)
plt.yscale('log')
for i in range(len(concata)):
    plt.plot(concata[i], label='submatrix size %s'%str(15-i))
plt.xlabel('position in the array: length is number of iterations') # because it is formed in the while k<maxit loop
plt.ylabel('$|T_{m,m-1}$|')
plt.title('A = D + O, '+shift)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()  # line plot

A2=1.0*A
Tmm1_array = script3e_dots(A2, 100, do_shift)
plt.yscale('log')
plt.plot(Tmm1_array, 'o')
plt.xlabel('iteration')
plt.ylabel('$|T_{m,m-1}$|')
plt.title('A = D + O, '+shift)
plt.show()  # scatter plot
