""" Your college id here: 01356309
    Template code for project 1, contains 5 functions:
    merge: used by func1A and func1B
    func1A and func1B: complete functions for question 1
    test_func1: function to be completed for question 1
    gene1: function to be completed for question 2
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt


def merge(L,R):
    """Merge 2 sorted lists provided as input
    into a single sorted list"""

    M = []  # Merged list, initially empty
    indL,indR = 0,0  # start indices
    nL,nR = len(L), len(R)

    #Add one element to M per iteration until an entire sublist
    #has been added
    for i in range(nL+nR):
        if L[indL] < R[indR]:
            M.append(L[indL])
            indL = indL + 1
            if indL >= nL:
                M.extend(R[indR:])
                break
        else:
            M.append(R[indR])
            indR = indR + 1
            if indR >= nR:
                M.extend(L[indL:])
                break
    # print(M)
    return M


def func1A(L_all):
    """Function to be analyzed for question 1
    Input: L_all: A list of N length-M lists. 
    Each element of L_all is a list of integers sorted in ascending order.
    Example input: L_all = [[1,3],[2,4],[6,7]], L_all = [[1,3],[6,7],[2,4]]
    """
    if len(L_all)==1:
        return L_all[0]
    else:
        L_all[-1] = merge(L_all[-2],L_all.pop())  # merge second-to-last and last elements into one list
        return func1A(L_all)


def func1B(L_all):
    """Function to be analyzed for question 1
    Input: L_all: A list of N length-M lists. Each element of  # note: the lists in the list don't have to be of the same length
    L_all is a list of integers sorted in ascending order.
    Example input with N=3,M=2: L_all = [[1,3],[2,4],[6,7]]
    """
    N = len(L_all)
    if N==1:
        return L_all[0]
    elif N==2:
        return merge(L_all[0],L_all[1])
    else:
        return merge(func1B(L_all[:N//2]),func1B(L_all[N//2:]))

    # for length-M 


def test_func1(inputs=None):
    """Question 1: Analyze performance of func1A and func1B
    Use the variables inputs and outputs if/as needed.
    You may import modules within this function as needed, please do not import
    modules elsewhere without permission.
    """

    # CHECKING N
    Nmax = 100
    l1_list_N = []
    l2_list_N = []
    for N in range(1, Nmax):
        M = 20  # doesnt change
        list_1 = []
        for j in range(N):
            list_s = []
            for i in range(M):
                n = random.randint(0,1000)
                list_s.append(n)  # append the random numbers to sublist
            list_s.sort()  # sort the sublist
            list_1.append(list_s)  # append the sublist to the list
        list_2 = list_1.copy()  # copy the list to use for func1B

        t1_1 = time.time()
        output1 = func1A(list_1)
        t1_2 = time.time()
        l1_list_N.append(t1_2-t1_1)

        t2_1 = time.time()
        output2 = func1B(list_2)
        t2_2 = time.time()
        l2_list_N.append(t2_2-t2_1)

    xaxis_N = np.arange(1,Nmax)

    # interpolation
    new_x = np.linspace(1,Nmax,30)
    coefs1_N = np.polyfit(xaxis_N,l1_list_N,2)  # quadratic
    new_line1_N = np.polyval(coefs1_N, new_x)

    coefs2_N = np.polyfit(xaxis_N,l2_list_N,1)  # linear
    new_line2_N = np.polyval(coefs2_N, new_x)

    plt.plot(xaxis_N, l1_list_N, label = "func1A")
    plt.plot(xaxis_N, l2_list_N, label = "func1B")
    plt.plot(new_x, new_line1_N, '-', label="quadratic interpolation, func1A")
    plt.plot(new_x, new_line2_N, '-', label="linear interpolation, func1B")
    plt.xlabel('N')
    plt.ylabel('dt')
    plt.title('Varying N, fixed M')
    plt.legend()
    plt.show()


    # CHECKING M
    Mmax = 75
    l1_list_M = []
    l2_list_M = []
    for M in range(1, Mmax):
        N = 200  # doesnt change

        list_1 = []
        for j in range(N):
            list_s = []
            for i in range(M):
                n = random.randint(0,1000)
                list_s.append(n)             # create a list
            list_s.sort()
            list_1.append(list_s)   
        list_2 = list_1.copy()  # copy the list to use for func1B

        t1_1 = time.time()
        output1 = func1A(list_1)
        t1_2 = time.time()
        l1_list_M.append(t1_2-t1_1)
        # print("dt1 = ",t1_2-t1_1)

        t2_1 = time.time()
        output2 = func1B(list_2)
        t2_2 = time.time()
        l2_list_M.append(t2_2-t2_1)
        # print("dt2 = ",t2_2-t2_1)
    xaxis_M = np.arange(1,Mmax)

    # interpolation
    new_x = np.linspace(1,Mmax,30)
    coefs1_M = np.polyfit(xaxis_M,l1_list_M,1)  # linear
    new_line1_M = np.polyval(coefs1_M, new_x)

    coefs2_M = np.polyfit(xaxis_M,l2_list_M,1)  # linear
    new_line2_M = np.polyval(coefs2_M, new_x)

    plt.plot(xaxis_M, l1_list_M, label = "func1A")
    plt.plot(xaxis_M, l2_list_M, label = "func1B")
    plt.plot(new_x, new_line1_M, '-', label="linear interpolation, func1A")
    plt.plot(new_x, new_line2_M, '-', label="linear interpolation, func1B")
    plt.xlabel('M')
    plt.ylabel('dt')
    plt.title('Varying M, fixed N')
    plt.legend()
    plt.show()


    # CHECKING N,M
    Nmax = 100
    l1_list_NM = []
    l2_list_NM = []
    for N in range(1, Nmax):
        M = N  # doesnt change
        list_1 = []
        for j in range(N):
            list_s = []
            for i in range(M):
                n = random.randint(0,1000)
                list_s.append(n)  # append the random numbers to sublist
            list_s.sort()  # sort the sublist
            list_1.append(list_s)  # append the sublist to the list
        list_2 = list_1.copy()  # copy the list to use for func1B

        t1_1 = time.time()
        output1 = func1A(list_1)
        t1_2 = time.time()
        l1_list_NM.append(t1_2-t1_1)

        t2_1 = time.time()
        output2 = func1B(list_2)
        t2_2 = time.time()
        l2_list_NM.append(t2_2-t2_1)

    xaxis_NM = np.arange(1,Nmax)

    # interpolation
    new_x = np.linspace(1,Nmax,30)
    coefs1_NM = np.polyfit(xaxis_NM,l1_list_NM,3)  # cubic
    new_line1_NM = np.polyval(coefs1_NM, new_x)

    coefs2_NM = np.polyfit(xaxis_NM,l2_list_NM,2)  # quadratic
    new_line2_NM = np.polyval(coefs2_NM, new_x)

    plt.plot(xaxis_NM, l1_list_NM, label = "func1A")
    plt.plot(xaxis_NM, l2_list_NM, label = "func1B")
    plt.plot(new_x, new_line1_NM, '-', label="cubic interpolation, func1A")
    plt.plot(new_x, new_line2_NM, '-', label="quadratic interpolation, func1B")
    plt.xlabel('N,M')
    plt.ylabel('dt')
    plt.title('Varying N,M')
    plt.legend()
    plt.show()


def char2base4(S):
    '''convert gene test sequance string in list of ints'''
    c2b = {}
    c2b['A'] = 0
    c2b['C'] = 1
    c2b['G'] = 2
    c2b['T'] = 3
    L = []
    for s in S:
        L.append(c2b[s])
    return L


def h_eval(L, base, prime):
    '''convert list L to base-10 number mod prime where base specifies the base of L'''
    f = 0
    for l in L[:-1]:
        f = base*(l+f)
    h = (f + (L[-1])) % prime
    return h


def gene1(S,L_in_arr,x):

    """Question 2: Complete function to find point-x mutations of patterns in
    gene sequence, S
    Input:
        S: String corresponding to a gene sequence
        L_in: List of P length-M strings. Each string corresponds to a gene
        sequence pattern, and M<<N
        x: integer setting location of point-x mutation (x<3)
    Output:
        L_out: List of lists containing locations of point-x mutations in S.
        L_out[i] should be a list of integers containing the locations in S at
        which all point-x mutations of the sequence in L_in[i] occur. If no
        mutations of L_in[i] are found, then L_out[i] should be empty.
    """

    N = len(S)  # length of sequence
    X = char2base4(S)  # convert to list of ints
    L_out = []
    for L_in in L_in_arr:
        M = len(L_in)  # length of pattern
        Y = char2base4(L_in)  # convert to list of ints
        print('base 4 L_in',Y)

        mutated = []  # get the (non-)mutations in a list
        if x==-1:
            mutated.append(Y)
            print('x is',x,', look for non-mutated string', mutated)
        else:
            Y1 = Y.copy()
            Y2 = Y.copy()
            Y3 = Y.copy()
            if Y[x]==0:
                Y1[x]=1
                mutated.append(Y1)
                Y2[x]=2
                mutated.append(Y2)
                Y3[x]=3
                mutated.append(Y3)
            if Y[x]==1:
                Y1[x]=0
                mutated.append(Y1)
                Y2[x]=2
                mutated.append(Y2)
                Y3[x]=3
                mutated.append(Y3)
            if Y[x]==2:
                Y1[x]=0
                mutated.append(Y1)
                Y2[x]=1
                mutated.append(Y2)
                Y3[x]=3
                mutated.append(Y3)
            if Y[x]==3:
                Y1[x]=0
                mutated.append(Y1)
                Y2[x]=1
                mutated.append(Y2)
                Y3[x]=2
                mutated.append(Y3)

        indmutations = []  # get the lists of matches of each mutation in one list indmutations
        base = 4  # choose base 4 because we have 4 letters
        prime = 9377  # choose an arbitrary large prime
        for i in range(len(mutated)):
            hp = h_eval(mutated[i],base,prime)  # create the hash for the mutation
            imatch = []
            hi = h_eval(X[:M], base, prime)  # create the hash for the first length-M substring of S
            if hi==hp:
                if X[:M]==mutated[i]:  # explicit check if the hash is the same
                    imatch.append(0)
                    print('mutation',mutated[i],'match at index ', 0)

            bm = (base**M) % prime
            for ind in range(1,N-M+1):  # update the hash function
                hi = (4*hi - int(X[ind-1])*bm + int(X[ind-1+M])) % prime
                if hi==hp:
                    if X[ind:ind+M]==mutated[i]:  # explicit check if the hash is the same
                        imatch.append(ind)
                        print('mutation',mutated[i],'match at index ', ind)  # found a match

            if len(imatch)==0:
                print('mutation',mutated[i],'no matches')  # no matches
            if len(imatch)!=0:
                indmutations.append(imatch)  # if matches, append to the indexes where we have mutations of L_in[i]
        
        print('indexes of mutations of', L_in, ':', indmutations,'\n\n')
        L_out.append(indmutations) # the indmutations - the list of where we have mutations of each L_in[i]
    print('\nL_out unsorted is ', L_out)

    L_outsort = []
    for l in L_out:
        if len(l)!=0:
            L_outsort.append(func1B(l))     # use func1B() to make each element of L_out a sorted list, decomposing
                                            # the three different lists that come from the three different mutations
        else:
            L_outsort.append(l)
    return L_outsort


def test_gene1_S():
    # increase length of S, elements of L_in constant, observe linear growth in computation time
    K = np.arange(100000-1,1000000,50000)
    time1 = []
    for k in K:
        t1 = time.time()
        L = gene1(S[:k],L_in_arr,x)
        t2 = time.time()
        time1.append(t2-t1)
    coefsK = np.polyfit(K,time1,1)  # linear
    new_lineK = np.polyval(coefsK, K)
    plt.plot(K, time1, 'x')
    plt.plot(K, new_lineK)
    plt.title('increase length of S, keep number of elements of L_in constant')
    plt.xlabel('K, where we apply gene1() to S[:K]')
    plt.ylabel('dt')
    plt.show()


def test_gene1_P():
    # increase P, the number of elements of L_in, observe linear growth in computation time
    l = 'GATGCTGA'
    K2 = 10
    L_in = []
    time3 = []
    for i in range(K2):
        L_in.append(l)
        t1 = time.time()
        L_out = gene1(S, L_in, 1)
        t2 = time.time()
        time3.append(t2-t1)
    x2 = np.arange(1,11)
    plt.plot(x2,time3,'x')
    plt.title('increase number of elements in L_in (elements are duplicates)')
    plt.xlabel('number of elements in L_in')
    plt.ylabel('dt')
    plt.show()


def test_gene1_SP():
    # increase N and P, observe quadratic growth in computation time
    K = np.arange(100000-1,1000000,100000)
    time4 = []
    l = 'GATGCTGA'
    L_in = []
    for k in K:
        L_in.append(l)
        t1 = time.time()
        L = gene1(S[:k],L_in,x)
        t2 = time.time()
        time4.append(t2-t1)

    coefsK = np.polyfit(K,time4,2)  # quadratic
    new_lineK = np.polyval(coefsK, K)
    plt.plot(K, time4, 'x')
    plt.plot(K, new_lineK, label='quadratic interpolation')
    plt.title('increase length N of S and number P of elements of L_in')
    plt.xlabel('K, where we apply gene1() to S[:K], P increases by 1 at each \'x\' ')
    plt.ylabel('dt')
    plt.legend()
    plt.show()


def test_gene1_M():
    # increase length of the elements of of L_in, keep S constant
    K1 = np.arange(5,10)
    time2 = []
    for k1 in K1:
        l_in_arr = []
        for lin in L_in_arr:
            l_in_arr.append(lin[:k1])
        t1 = time.time()
        L = gene1(S,l_in_arr,x)
        t2 = time.time()
        time2.append(t2-t1)
    plt.plot(K1, time2)
    plt.title('increase length of the elements in L_in, fixed length of S')
    plt.xlabel('K, where we apply gene1() to L_in with elements of increasing length L_in[i][:K]')
    plt.ylabel('dt')
    plt.show()


if __name__=='__main__':
    inputs=None

    # output plots for question 1
    test_func1()
    

    # question 2
    infile = open('Sexample.txt','r')
    S = infile.read()  # read string Sexample, store in S
    infile.close()
    print('length of S is ',len(S))

    L_in_arr = ['GAGATTCAAG', 'TAGTCGATCA', 'CCTGAGCTAG']
    x = 0
    # # uncomment to test one run
    # L_out = gene1(S,L_in_arr,x)
    # print('\nfinal sorted L_out is',L_out)

    # output plots for question 2 (may take a few minutes)

    # increase length N of S, elements of L_in constant, observe linear growth in computation time
    test_gene1_S()

    # increase P, the number of elements of L_in, length of S constant, observe linear growth in computation time
    test_gene1_P()

    # increase N and P, observe quadratic growth in computation time
    test_gene1_SP()

    # increase length M of the elements of of L_in, keep N and P constant
    test_gene1_M()
