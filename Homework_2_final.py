#Homework 2
#Joey Latta

import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt


# Problem A: multiples of 3 or 5 (Euler 1)
def mult_3_5():
    print("Input a number to find the sum of all multiples of 3 or 5 below that number: ")
    total_sum = int(input())
    final_sum = 0
    for  i in range(1,total_sum):
        if (i % 3 == 0 or i % 5 == 0):
            final_sum = final_sum + i
    return final_sum
print("\n", mult_3_5())

# Problem B: largest prime factor (Euler 3)
def largest_prime_factor():
    print("Input a number to find the largest prime factor of that number: ")
    n = int(input())
    i = 2
    while i * i <= n:
        if n % i: #if the remainder is not zero...
            i += 1
        else:
            n //= i #floor division
    return n
print("\n", largest_prime_factor())

# Problem C: smallest multiple (Euler 5)
def small_mult():
    print("Input a number to find the smallest positive number that is evenly divisible by all numbers from one to that number: ")
    m = int(input())
    i = 1
    for k in (range(1, m+1)):
        if i % k > 0:
            for j in range(1, m+1):
                if (i*j) % k == 0:
                    i *= j
                    break
    return i
print("\n", small_mult())

# Problem D: 10001st prime number (Euler 7)

def primes_sieve():
    print("Find the xth prime number: ")
    n = int(input())
    p_n = int(2 * n * math.log(n))       # over-estimate p_n
    sieve = [True] * p_n                 # everything is prime to start
    count = 0
    for i in range(2, p_n):
        if sieve[i]:                       # still prime?
            count += 1                     # count it!
            if count == n:                 # done!
                return i
            for j in range(2*i, p_n, i):   # cross off all multiples of i
                sieve[j] = False
    return j
print("\n", primes_sieve())

# Problem E: special Pythagorean triplet (Euler 9)
def pyth_triple():
    print("Input a number to find the Pythagorean triplet of that number: ")
    n = int(input())
    for a in range(1, n):
        for b in range(a, n):
         c = n - a - b
         if c > 0:
             if c*c == a*a + b*b:
                x = (a*b*c)
                break
    return x
print("\n", pyth_triple())


# Problem 6: matrices and vectors

# Matrix multiplied by a vector
def matrix_x_vector():
    n = 3
    m = 5
    a = np.array(np.random.random((m, n)))
    b = np.array(np.random.random((n,1)))
    matr_x_vector = np.dot(a, b) # I used loops to multiply matrices in next part so I figured I could try .dot here
    print("Matrix a times vector b = \n",matr_x_vector)
matrix_x_vector()

# Square matrix multiplication
def square_matrix_mult():

    #establish matrix dimensions
    n = 3
    m = 500

    #creat square matrices filled with random numbers
    a = np.array(np.random.random((n,n)))
    b = np.array(np.random.random((n,n)))
    c = np.array(np.random.random((m,m)))
    d = np.array(np.random.random((m,m)))

    #Row-major Order

    # Multiply nxn matrix in row-major order and time the calculation
    tick1 = time.time()
    ans=[[0]*n for i in range(n)]

    #iterate through rows of a
    for i in range(n):
        #iterate through columns of b
        for j in range(n):
            #fill in matrix in row-major order
            ans[i][j]=sum((a[i][v]*b[v][j] for v in range(n)))
    print("\n (by row-major order) matrix a times matrix b = \n",ans)
    tock1 = time.time()
    time_delta_1 = float(tock1 - tick1)
    print("\n Row-major order took {} s to multiply two {}x{} matrices: ".format(time_delta_1, n, n))

    # Multiply mxm matrix in row-major order and time the calculation
    tick2 = time.time()
    ans2=[[0]*m for i in range(m)]
    for i in range(m):
        for j in range(m):
            ans2[i][j]=sum((c[i][v]*d[v][j] for v in range(m)))
    #print(ans2)
    tock2 = time.time()
    time_delta_2 = float(tock2 - tick2)
    print("\n Row-major order took {} s to multiply two {}x{} matrices: ".format(time_delta_2, m, m))

# Column Major Order
    # Multiply nxn matrix in column-major order and time the calculation
    tick3 = time.time()
    ans3=[[0]*n for i in range(n)]
    for i in range(n):
        for j in range(n):
            #fill in matrix in column-major order
            ans3[j][i]=sum((a[j][v]*b[v][i] for v in range(n)))
    print("\n (by column-major order) matrix a times matrix b = \n", ans3)
    tock3 = time.time()
    time_delta_3 = float(tock3 - tick3)
    print("\n Column-major order took {} s to multiply two {}x{} matrices: ".format(time_delta_3, n, n))

    # Multiply mxm matrix in column-major order and time the calculation
    tick4 = time.time()
    ans4=[[0]*m for i in range(m)]
    for i in range(m):
        for j in range(m):
            ans4[j][i]=sum((c[j][v]*d[v][i] for v in range(m)))
    #print(ans4)
    tock4 = time.time()
    time_delta_4 = float(tock4 - tick4)
    print("\n Column-major order took {} s to multiply two {}x{} matrices: ".format(time_delta_4, m, m))

square_matrix_mult()

def gen_matrix_mult():
    # Multiply a kxl matrix times a lxm matrix

    # establish matrix dimenstions k, l, m
    # note: could have these user input but for not necessary for quick demonstration
    k = 3
    l = 4
    m = 5

    # create a kxl and lxm matrix filled with random numbers
    e = np.array(np.random.random((k,l)))
    f = np.array(np.random.random((l,m)))

    # create kxm resultant matrix to fill in
    ans= np.zeros((k,m))

    # iterate through rows of e
    for i in range(len(e)):
        #iterate through columns of f
        for j in range(len(f[0])):
            #iterate through rows of f
            for kk in range(len(f)):
                ans[i][j] += e[i][kk] * f[kk][j]
    print("\n matrix e (3x4) times matrix f (4x5) = \n", ans)
gen_matrix_mult()

# Problem 7: Driven harmonic oscillator vs amplitude modulated oscillator
def oscillator():

    steps = 10000 # change to make smaller time step

    finaltime = 100

    dt = finaltime/steps

    print("Time step is", dt)

    time = np.linspace(0, finaltime, steps + 1.0) #initiate time

    v = np.zeros(steps + 1) # initiate velocity data

    x = np.zeros(steps + 1) # initiate displacement data

    x[0] = 0.0

    v[0] = 0.1

    print("Space is: ", x)

    print("The initial displacement = {}m and the initial velocity = {}m/s".format(x[0],v[0]))

    a = 1.0

    b = 1

    q = 0.0

    #discretize to solve ODE
    for k in range(0, steps):
        #first determine v at each time step
        v[k+1] = v[k] - (a - 2*q * np.cos( k * dt * 2.0)) * dt * x[k]
        #determine next displacement to loop
        x[k+1] = x[k] + dt * v[k]

    plt.figure(1)

    plt.plot(time, x, 'g-', label = 'x(t)')

    plt.show()

    for kk in range(0, steps):

        v[kk+1] = v[kk] - a * dt * x[kk]

        x[kk+1] = x[kk] + dt + v[kk]

    plt.figure(2)

    plt.plot(time, x, 'g-', label = 'x(t)')

    plt.show()


oscillator()
