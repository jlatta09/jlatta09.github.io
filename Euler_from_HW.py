## Joey Latta
## These are the Euler problems from assigned HW
## I just copy fasted these functions from HW to this file so the file as a
## whole may not run, but each function should be able to run indepentantly

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

# A: Largest Product in a grid (Euler 11)
def largeProduct():

    grid=[[8,2,22,97,38,15,0,40,0,75,4,5,7,78,52,12,50,77,91,8],
    [49,49,99,40,17,81,18,57,60,87,17,40,98,43,69,48,4,56,62,0],
    [81,49,31,73,55,79,14,29,93,71,40,67,53,88,30,3,49,13,36,65],
    [52,70,95,23,4,60,11,42,69,24,68,56,1,32,56,71,37,2,36,91],
    [22,31,16,71,51,67,63,89,41,92,36,54,22,40,40,28,66,33,13,80],
    [24,47,32,60,99,3,45,2,44,75,33,53,78,36,84,20,35,17,12,50],
    [32,98,81,28,64,23,67,10,26,38,40,67,59,54,70,66,18,38,64,70],
    [67,26,20,68,2,62,12,20,95,63,94,39,63,8,40,91,66,49,94,21],
    [24,55,58,5,66,73,99,26,97,17,78,78,96,83,14,88,34,89,63,72],
    [21,36,23,9,75,0,76,44,20,45,35,14,0,61,33,97,34,31,33,95],
    [78,17,53,28,22,75,31,67,15,94,3,80,4,62,16,14,9,53,56,92],
    [16,39,5,42,96,35,31,47,55,58,88,24,0,17,54,24,36,29,85,57],
    [86,56,0,48,35,71,89,7,5,44,44,37,44,60,21,58,51,54,17,58],
    [19,80,81,68,5,94,47,69,28,73,92,13,86,52,17,77,4,89,55,40],
    [4,52,8,83,97,35,99,16,7,97,57,32,16,26,26,79,33,27,98,66],
    [88,36,68,87,57,62,20,72,3,46,33,67,46,55,12,32,63,93,53,69],
    [4,42,16,73,38,25,39,11,24,94,72,18,8,46,29,32,40,62,76,36],
    [20,69,36,41,72,30,23,88,34,62,99,69,82,67,59,85,74,4,36,16],
    [20,73,35,29,78,31,90,1,74,31,49,71,48,86,81,16,23,57,5,54],
    [1,70,54,71,83,51,54,69,16,92,33,48,61,43,52,1,89,19,67,48]]

    largest=[0,0,0,0]

    #horizontal and vertical
    for h in range(0,20):
        for hsub in range(0,17):
            horizontal = grid[h][hsub] * grid[h][hsub+1] * grid[h][hsub+2] * grid[h][hsub+3]
            vertical   = grid[hsub][h] * grid[hsub+1][h] * grid[hsub+2][h] * grid[hsub+3][h]
            if horizontal > largest[0]:
                largest[0] = horizontal
            if vertical > largest[1]:
                largest[1] = vertical

    #diagonal right and left
    for r in range(0,17):
        for rsub in range (0,17):
            right_diagonal = grid[rsub][0+r] * grid[rsub+1][1+r] * grid[rsub+2][2+r] * grid[rsub+3][3+r]
            left_diagonal  = grid[rsub][3+r] * grid[rsub+1][2+r] * grid[rsub+2][1+r] * grid[rsub+3][r]
            if right_diagonal > largest[2]:
                largest[2] = right_diagonal
            if left_diagonal > largest[3]:
                largest[3] = left_diagonal

    print("The greatest product of four adjacent numbers in the 20x20 array is: ", max(largest))

largeProduct()


# B: Power Digit Sum (Euler 16)
def power2sum():

    total = 2**1000
    total = str(total)
    result = 0
    for i in range(len(total)):
        result += int(total[i])
    print("The sum of the digits in the number 2^1000 is: ", result)

power2sum()

# C: factorial digit sum (Euler 20)
def factorial(n):
    if n == 0: return(1)
    else: return(n * factorial(n-1))


def main():
    x = 100
    answer = factorial(x)
    digits = list(str(answer))

    sum_dig = 0
    for digit in digits:
        sum_dig += int(digit)
    print("The sum of the digits of 100! is: ", sum_dig)
main()

# D: 1000-digit Fibonacci number (Euler 25)
def fibb():
    index = 2
    fib_list = [1,1]
    while len(str(fib_list[-1])) < 1000:
        fib_list.append(fib_list[-1] + fib_list[-2])
        index += 1
    print("The index of the first term in the Fibonacci sequence is: ", index)
fibb()

#Problem A: Reciprocal Cycles (Euler 26)

def rec_cycle(d):

    results = {} # create empty set to store results 1/d
    n = 1 # set numerator = 1
    dec = 0 # track decimal place

    # iterate up to denominator limit d, -> store in results
    # -> convert results set to only look at decimal places
    while n:
        if n < d:
            dec += 1
            n *= 10
        else:
            n %= d
            if n in results:
                return dec - results[n]
            results[n] = dec
    return 0

ans = (0, 0)
# find the maximum length of recurring cycle
for i in range(2,1000):
    ans = max(ans, (rec_cycle(i), i)) # manually implemented max in previous hw, used the python function here

print("The value d < 1000 for which 1/d contains the longest recurring cycle in its decimal fraction part is:\n{}".format(ans[1]), "\n")


# Problem B: Quadratic Primes (Euler 27)

def is_prime(x): # function to return prime numbers

    if x<2:return 0
    if x%2:
        for i in range(3,int(np.sqrt(x)),2):
            if x%i==0:return 0
    else:return 0
    return 1

def product_of_primes(): # function to solve f(x) and return product

    product,n=41,40 # initiate n at 41

    for a in range(-999,1000):
        for b in range(0,1000):
            c=0
            while is_prime(c*c+a*c+b)==1:c+=1 # check if prime number, if so continue next step
            if c>n:
                n,product=c,a*b
    print("The product of a and b in f(n)=n^2 + an + b for Euler 27 is:\n{}".format(product), "\n")
product_of_primes()

# Problem C: Number Spiral Diagonals (Euler 28)
def spiral(num):
    diagSum = 1 # start spiral at 1
    dist = 2 # the difference between corner numbers grows by 2 for each 2 rows added

    # start at the top right corner of the first square
    # calculate the other 3 corners of that square and sum them
    # iterate until through the num x num spiral
    for i in range(3, num+1, 2):
            for j in range(4):
                diagSum += i**2 - dist * j
            dist += 2
    return diagSum
print("The sum of numbers on the diagonals of the 1001x1001 spiral is:\n{}".format(spiral(1001)), "\n")

# Problem A: Digit fifth powers (Euler 30)


def digitPower(n):

        Sum = 0
        # iterate up to max
        # max number to loop to = (9^n)*n

        for i in range(2,9**n*n):

                s = 0
                # iterate over i each number^5
                for j in str(i):
                        s += int(j)**n
                # pull out numbers to sum
                if s == i:
                        Sum += s
        print(Sum)
digitPower(5)

# Problem B: Pandigital Product (Euler 32)

def pandigital():

    prodSet = set() # create an empty set to store products without duplicates
    pandig9 = set("123456789") # create a 1-9 pandigital set to check against

    # iterate 0-9 for fisrt multiplicand and 999-9999 multiplyer
    for i in range(9):

        for j in range(999,9999):
            #convert to strings and add multiplicand, multiplyer and product
            p = str(i) + str(j) + str(i*j)
            # check if 1-9 pandigital
            if len(p) == 9 and set(p) == pandig9:
                prodSet.add(i*j)

            elif len(p) > 9: break
    # iterate 9-99 for fisrt multiplicand and 99-999 multiplyer
    for i in range(9,99):

        for j in range(99,999):
            p = str(i) + str(j) + str(i*j)

            if len(p) == 9 and set(p) == pandig9:
                prodSet.add(i*j)

            elif len(p) > 9: break

    print(sum(prodSet))
pandigital()

 Problem I: Double-Base Palindromes (Euler 36)

# converts decimal to binary
def dec_to_bin(num):

    binary = ''

    while num > 0:

        binary = str(num % 2) + binary
        num = num >> 1

    return binary

# print(dec_to_bin(24)) # test dec_to_bin

def check_pal():

    sum = 0

    # only need to check odd numbers
    for i in range(1,1000000,2):
        # first check if decimal number is palindrome
        if str(i) == str(i)[::-1]:
            # next check if binary is palindrome
            if dec_to_bin(i) == dec_to_bin(i)[::-1]:
                sum += i

    print("The sum of all numbers, less than one million, which are palindromic in base 10 and base 2 is: ", sum)

check_pal()

# Problem II: Integer Right Triangles (Euler 39)

def triangle(maxp):

   solns = [0] * int(maxp+1)

   # iterate a and b from 1 to max perimeter (only need evens)
   for a in range(2, maxp+1, 2):

      for b in range(2, maxp+1, 2):

         c = np.sqrt(a*a + b*b)

         # get rid of invalid solutions
         if a+b+c > maxp:

            continue

         if c - int(c) > 0:

            continue

        # only calculate p for integer right triangles
         c = int(c)
         p = a+b+c
         solns[p] = solns[p] + 1

   print (solns.index(max(solns)))
triangle(1000)
