#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:10:54 2024

@author: maryamramezaniziarani
"""
#%%
#1. Write a Python function, is_factor(n, m), that takes two integer values and
#returns True if n is a factor of m, and False otherwise.

def is_factor(n, m):
    if n == 0:
        return False  
    if m % n == 0:
        return True  
    else:
        return False  
#%%
#2. Write a Python function, is_odd(k), that takes an integer value and returns
#True if k is an odd number, and False otherwise. However, your function can 
#not use the multiplication, modulo, or division operators.

def is_odd(k):
    """
    Return True if k is odd, otherwise return False (without using *, %, or /).
    """
    
    if k & 1 == 1:
        return True  
    else:
        return False  
#%%
#3. Write a Python function called calculate_factorial(n) that calculates the
#factorial of a non-negative integer, n, using a for loop. 

def calculate_factorial(n):
    """
    Calculate the factorial of a non-negative integer n.
    """
    if n < 0:
        return False  
    if n == 0:
        return 1  
    fac = 1  
    for i in range(1, n + 1):
        fac *= i  
    return fac  
#%%
#4. A number is a palindrome if the $i$-th decimal digit from the front
# coincides  with the $i$-th decimal digit from the back.
#E.g., 18781 is a palindrome, whereas 84736 is not.
#Write a function is_palindrome(n) that returns True if $n$ is a palindrome, and False otherwise. 

def is_palindrome(n):
    """
    Return True if n is a palindrome, otherwise return False.
    """
    
    n_str = str(n)
    
    for i in range(len(n_str) // 2):
        if n_str[i] != n_str[-i - 1]:
            return False  
    else:    
        return True  
#%%
#1. Write a function `gcd(a,b)` which implements the Euclidean algorithm for computing
#the greatest common divisor of two positive integers $a$ and $b$.

def gcd(a,b):
    while b != 0:
        (a, b) = (b, a % b)
    return a

# Example 
result = gcd(48, 18)
print("GCD:", result)  
#%%
#2. Write a function `machine_epsilon()` which returns the machine epsilon of the built-in 
#floating point representation.  The machine epsilon is defined as follows: it is the smallest 
#number $\epsilon$ such that $1 + \epsilon > 1$ when computed in the machine floating point representation.

def machine_epsilon():
        x =1  
        while x+1>1 :     
            x/=2
        return(2*x) 

epsilon = machine_epsilon()
print("Machine epsilon:", epsilon)
#%%
#3. Write a function `my_reverse(L)` that takes a list `L` and returns the list in reverse order.
#Do not use the builtin `reverse()` method.

def my_reverse(L):
    return L[::-1]

# Example 
list1 = [1, 2, 3, 4, 5]
reversed_list1 = my_reverse(list1)
print("Original list1:", list1)
print("Reversed list1:", reversed_list1)
#%%
#4. Write a function `powers_of_2(n)` which returns a list containing the numbers $2^0, \ldots, 2^{n-1}$.
#Your code must not contain more than 55 characters.

def powers_of_2(n):
    return [2**i for i in range(n)]
#%%
#1. Write a Python function `make_change(charged,given)` that "makes change". 
#The function should take two integers as arguments: the first represents the monetary 
#amount charged, and the second is the amount given, both in Euro cents. The function should 
#return a Python dictionary containing the Euro bills or coins to give back as change between
#the amount given and the amount charged. Design your program to return as few bills and coins
#as possible.

#For example, make_change(6705, 10000) should return
#{20.0: 1, 10.0: 1, 2.0: 1, 0.5: 1, 0.2: 2, 0.05: 1}.

#If the provided inputs are not integers or if the amount given is less than the amount charged,
#the function should raise a ValueError with an appropriate error messages;
#"Both 'charged' and 'given' should be integers." and "Too little money given" .

def make_change(charged, given):
    try:
        if not isinstance(charged, int) or not isinstance(given, int):
            raise TypeError("Both 'charged' and 'given' should be integers.")

        if given < charged:
            raise ValueError("Too little money given")

        change = given - charged
        denominations = [2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]
        change_dict = {}

        for denom in denominations:
            count = change // denom
            if count > 0:
                change_dict[denom / 100.0] = int(count)
                change %= denom

        return change_dict
    except TypeError as e:
        raise ValueError(str(e))

# Example 
result = make_change(6705, 9500)
print(result)  
#%%
#2. Write a short Python function `has_odd_product(L)` that takes a 
#list of integer values `L` and determines if there is a distinct pair of numbers in the 
#sequence whose product is odd.

def has_odd_product(L):
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if (L[i] * L[j]) % 2 == 1:
                return True
    return False 

list1 = [1, 2, 3, 4, 5]
result1 = has_odd_product(list1)
print("List1:", list1, "Has odd product:", result1)  
#%%
#3. Write a Python function `are_distinct(L)` that takes a list of numbers
#`L` and determines if all the numbers are different from each other (that is, they are distinct).

def are_distinct(L):
    return len(L) == len(set(L))
#%%
#1. Write a function `to_float(d)` which takes a decimal fraction as a string `d` and converts it to decimal
#floating point representation by returning a tuple `(m,e)` containing the integer significant $s$ and an
#exponent $e$ such that the number is given by $s \cdot 10^e$. To make the floating point representation 
#unique, we require that $s$ does not contain the digit 0 in the last place of its decimal representation.
#For simplicity, you may assume that the number is positive.For example, `to_float("123.45")` should return
#`(12345,-2)` while `to_float("99000")` should return `(99,3)`.

def to_float(s):
    s = s.strip()

    if "." in s:
        a, b = s.split(".")
        b = b.rstrip("0")
        return (int(a + b), -len(b))
    else:
        s_without_trailing_zero = s.rstrip("0")
        return (int(s_without_trailing_zero), len(s) - len(s_without_trailing_zero))

# Example 
result = to_float("123.45000")
print(result)
#%%
#2. Write a function `format_table(s)` which takes a string containing unstructured expense information, 
#separated by comma, such as

#    ```
#   Dinner with Ted, 30.00, Bus back home,  2, Present for aunt, Mary 25.99
#    ```
#You should return a string which contains the same data as a formatted table, plus a final row for the 
#total amount, which should print as shown:

#    ```
#    Dinner with Ted       30.99
#    Bus back home          2.00
#    Present for aunt Mary 25.99
#    TOTAL                 57.99
#    ```
#Note: to start a new line, add a newline character `"\n"` to the output string.

def format_table(s: str):
    separated = s.split(",")
    structured = [(e.strip(), f'{float(separated[i + 1]):.2f}') 
                  for i, e in enumerate(separated) if not i & 1]
    formatted, total = "", 0
    char_lim = max(max(len(k) for k, v in structured), 5) + max(len(v) for k, v in structured) + 1

    for k, v in structured:
        formatted += k + " " * (char_lim - len(k + v)) + v + "\n"
        total += float(v)

    return formatted + f'TOTAL{" " * (char_lim - 5 - len(str(round(total, 2))))}{round(total, 2)}'

# Example
print(format_table("Dinner with Ted, 30.00, Bus back home, 2, Present for aunt Mary, 25.99"))
print(format_table("Ted, -34, Mary, 45.995"))
#%%
#3. Write a function `change_key(D, old, new)` which changes key `old` to a new key `new` in the dictionary.
#If key `old` is not contained in the dictionary, the function does nothing.
    
def change_key(D, old, new):
    if old in D:
        D[new] = D.pop(old)
        
# Example 
my_dict = {"key1": "value1", "key2": "value2"}
change_key(my_dict, "key1", "new_key")
print(my_dict)
#%%
#1. (From GTG R-2.5.) The file `credit_card.py` contains the code of the credit card example from the book.  
#Revise the `charge` and `make_payment` methods to raise a `ValueError` if the caller does not send a number.
class CreditCard:
    """A consumer credit card."""

    def __init__(self, customer, bank, acnt, limit):
        """Create a new credit card instance.

        The initial balance is zero.

        customer  the name of the customer (e.g., 'John Bowman')
        bank      the name of the bank (e.g., 'California Savings')
        acnt      the acount identifier (e.g., '5391 0375 9387 5309')
        limit     credit limit (measured in dollars)
        """
        self._customer = customer  # Customer's name
        self._bank = bank          # Bank's name
        self._account = acnt       # Account number
        self._limit = limit        # Credit limit
        self._balance = 0          # Initial balance set to 0

    def get_customer(self):
        """Return name of the customer."""
        return self._customer
    
    def get_bank(self):
        """Return the bank's name."""
        return self._bank

    def get_account(self):
        """Return the card identifying number (typically stored as a string)."""
        return self._account

    def get_limit(self):
        """Return current credit limit."""
        return self._limit

    def get_balance(self):
        """Return current balance."""
        return self._balance

    def charge(self, price):
        """Charge given price to the card, assuming sufficient credit limit.

        Return True if charge was processed; False if charge was denied.
        """
        ## Check if the input price is a number (int or float)
        if not isinstance(price, (int, float)):
            raise ValueError("Price must be a number.")

        if price + self._balance > self._limit:  
            return False  
        else:
            self._balance += price  
            return True

    def make_payment(self, amount):
        """Process customer payment that reduces balance."""
        ## Check if the input amount is a number (int or float)
        if not isinstance(amount, (int, float)):
            raise ValueError("Amount must be a number.")

        
        self._balance -= amount

if __name__ == '__main__':
    wallet = []
    wallet.append(CreditCard('John Bowman', 'California Savings',
                             '5391 0375 9387 5309', 2500))
    wallet.append(CreditCard('John Bowman', 'California Federal',
                             '3485 0399 3395 1954', 3500))
    wallet.append(CreditCard('John Bowman', 'California Finance',
                             '5391 0375 9387 5309', 5000))

    for val in range(1, 17):
        wallet[0].charge(val)         
        wallet[1].charge(2*val)
        wallet[2].charge(3*val)

    for c in range(3):
        
        print('Customer =', wallet[c].get_customer())
        print('Bank =', wallet[c].get_bank())
        print('Account =', wallet[c].get_account())
        print('Limit =', wallet[c].get_limit())
        print('Balance =', wallet[c].get_balance())
        
        while wallet[c].get_balance() > 100:
            wallet[c].make_payment(100)
            print('New balance =', wallet[c].get_balance())
        print()
#%%
#2. (From GTG R-2.9ff.) The file `vector.py` contains the code of the vector class from the book.  Add the following 
#methods to the class: `__sub__` so that the expression `u-v` returns a new vector instance representing the difference 
#between two vectors and `__neg__` so that the expression `-v` returns a new vector instance whose coordinates are all 
#the negated values of the respective coordinates of `v`.  If the vectors do not have the same dimension, raise an error 
#as for `__add__` in the example code.

#3. Continuing Problem 2: Implement the `__mul__` method so that `u*v` returns a scalar that represents the dot product 
#of the vectors `u` and `v`, i.e., $u_1 \, v_1 + \cdots + u_d \, v_d$, and that `u*a` results in scalar multiplication 
#if `a` is a number.  Further, implement the `__rmul__` method to make sure that `a*v` is the same as `v*a` when `v` is 
#a vector and `a` is a number.

#4. Continuing Problem 3: Implement the method `cross` so that `u.cross(v)` gives the cross product of the vectors `u` 
#and `v` if both of their length is 3.  Raise a value error otherwise.

#Copyright 2013, Michael H. Goldwasser
#
# Developed for use with the book:
#
#    Data Structures and Algorithms in Python
#    Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser
#    John Wiley & Sons, 2013
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import collections
class Vector:
    """Represent a vector in a multidimensional space."""

    def __init__(self, d):
        
        if isinstance(d, int):
            self._coords = [0] * d  
        else:
            try:
                
                self._coords = [val for val in d]
            except TypeError:
                
                raise TypeError('invalid parameter type')

    def __len__(self):
        """Return the dimension of the vector."""
        
        return len(self._coords)

    def __getitem__(self, j):
        """Return jth coordinate of vector."""
        
        return self._coords[j]

    def __setitem__(self, j, val):
        """Set jth coordinate of vector to given value."""
        
        self._coords[j] = val

    def __add__(self, other):
        """Return sum of two vectors."""
        
        if len(self) != len(other):
            
            raise ValueError('dimensions must agree')
        result = Vector(len(self))  
        for j in range(len(self)):
            
            result[j] = self[j] + other[j]
        return result

    def __eq__(self, other):
        """Return True if vector has same coordinates as other."""
        
        return self._coords == other._coords

    def __ne__(self, other):
        """Return True if vector differs from other."""
        
        return not self == other

    def __str__(self):
        """Produce string representation of vector."""
        
        return '<' + str(self._coords)[1:-1] + '>'

    def __lt__(self, other):
        """Compare vectors based on lexicographical order."""
        
        if len(self) != len(other):
            raise ValueError('dimensions must agree')
        return self._coords < other._coords

    def __le__(self, other):
        """Compare vectors based on lexicographical order."""
        
        if len(self) != len(other):
            raise ValueError('dimensions must agree')
        return self._coords <= other._coords

    # New methods added below

    def __sub__(self, other):
        """Return the difference of two vectors."""
        # Subtract one vector from another.
        if len(self) != len(other):
            raise ValueError('dimensions must agree')
        result = Vector(len(self))  
        for j in range(len(self)):
            
            result[j] = self[j] - other[j]
        return result

    def __neg__(self):
        """Return the negation of the vector."""
        # Negate the vector
        result = Vector(len(self))
        for j in range(len(self)):
            result[j] = -self[j]
        return result

    def __mul__(self, other):
        """Return the dot product or scalar multiplication."""
        # Implement both the dot product and scalar multiplication.
        try:
            
            _ = iter(other), iter(self)
            if len(self) != len(other):
                raise ValueError('dimensions must agree')
            
            return sum(e1 * e2 for e1, e2 in zip(self, other))
        except TypeError:
            
            return Vector(e * other for e in self)

    def __rmul__(self, other):
        """Return the scalar multiplication."""
        # Support scalar multiplication when the scalar is on the left of the vector.
        return self.__mul__(other)

    def cross(self, vector: "Vector"):
        """Return the cross product of two vectors."""
        # Calculate the cross product of two 3D vectors.
        if not len(self) == len(vector) == 3:
            
            raise ValueError("Unmet conditions!")
        
        w = [self[1] * vector[2] - self[2] * vector[1],
             self[2] * vector[0] - self[0] * vector[2],
             self[0] * vector[1] - self[1] * vector[0]]
        return Vector(w)

def cross(vector_1: Vector, vector_2: Vector):
    """Return the cross product of two vectors."""
    return vector_1.cross(vector_2)


if __name__ == '__main__':
  # the following demonstrates usage of a few methods
  v = Vector(5)              # construct five-dimensional <0, 0, 0, 0, 0>
  v[1] = 23                  # <0, 23, 0, 0, 0> (based on use of __setitem__)
  v[-1] = 45                 # <0, 23, 0, 0, 45> (also via __setitem__)
  print(v[4])                # print 45 (via __getitem__)
  u = v + v                  # <0, 46, 0, 0, 90> (via __add__)
  print(u)                   # print <0, 46, 0, 0, 90>
  total = 0
  for entry in v:            # implicit iteration via __len__ and __getitem__
    total += entry
#%%
#1. Implement a function named `transform_elements`. This function takes as input an integer NumPy array `np_input_array`
#and an additional parameter `threshold`. The function should return a new NumPy array where each element less than the threshold 
#is squared,and each even element greater than or equal to the threshold is doubled.

import numpy as np

def transform_elements(np_input_array, threshold):
    transformed_array = np.where(np_input_array < threshold, np_input_array**2, np_input_array)
    transformed_array = np.where((np_input_array >= threshold) & (np_input_array % 2 == 0), 2 * np_input_array, transformed_array)
    return transformed_array
#%%
#2. Write a function `get_primes_up_to(n)` which takes as input a natural number $n$ and returns a numpy array containing all 
#the prime numbers smaller or equal to $n$.

import numpy as np

def get_primes_up_to(n):
    primes = np.arange(2, n+1)
    for i in range(n-1):
        p = primes[i]
        if p != 0:
            primes[(primes > p) & (primes % p == 0)] = 0
    return primes[primes != 0]
#%%
#3. Write a function `get_sum_of_composites_and_primes_squared_up_to(n)`that takes as input a natural number $n$ and returns the 
#sum of all composite (non-prime) numbers smaller or equal to $n$ plus the sum of squared primes smaller or equal to $n$.Consider 
#importing the `get_primes_up_to(n)` function from Problem2. Since the prime numbers follow no known regular pattern, a basic 
#indexing solution to this problem is not easy. Indexing by an array instead of by slicing can prove useful.

import numpy as np

def get_sum_of_composites_and_primes_squared_up_to(n):
    primes = get_primes_up_to(n)
    total_sum = n * (n + 1) / 2
    sum_of_primes = np.sum(primes)
    sum_of_primes_squared = np.sum(primes**2)
    return total_sum - sum_of_primes + sum_of_primes_squared
#%%
#1. Write a function `find_root_of_poly(coeffs, a, b, tol)` that approximates a root of a degree $3$ polynomial contained in the 
#interval $[a,b]$ (the input polynomial your function will be tested on will indeed have a real root in the interval). 
#The approximation is supposed to be within `tol` precision. The `coeffs` input will be a list of the coefficients of the 
#polynomial, starting with the coefficient of the free term and finishing with the coefficient of $x^3$.

import numpy as np
from numpy.polynomial import Polynomial

def find_root_of_poly(coeffs, a, b, tol):
    left = a
    right = b
    half = (b + a) / 2
    steps = int(np.log2((b - a) / tol))
    p = Polynomial(coeffs)
    
    for i in range(steps):
        if p(left) * p(half) < 0:
            right = half
        else:
            left = half
        half = (right + left) / 2
        
    return half
#%%
#2. The file contains data point arrays $x$ and $y$ which come from a power-law dependency $x\mapsto cx^\alpha$. 
#First do LSE to obtain the coefficients $c, \alpha$. Write a function `estimate_value(in_val)` that evaluates the obtained 
#power-law at an arbitrary input `in_val`.

import numpy as np
from numpy.polynomial import Polynomial

# Provided data points
x = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
y = np.array([3, 4, 5, 7, 9, 13, 15, 19, 23, 24, 29, 38, 40, 50, 56, 59, 70, 89, 104, 130])


log_x = np.log(x)
log_y = np.log(y)


p = Polynomial.fit(log_x, log_y, deg=1)

def estimate_value(in_val):
    return np.exp(p(np.log(in_val)))
#%%
#3. Using the data from Problem 2, plot the given data points $x_i, y_i$ as green marbles and the LSE solution as a red curve. 
#Also plot the predicted value as a larger blue marble and extend the domain of the LSE solution as a doted red curve from 
#last data input coordinate `x[-1]` to `in_val`. You should assume that `in_val = 30`. You are required to produce a single plot 
#with the described features. The plot should also have axis labels $x$ respectively $y$. 
#To save the figure, call `plt.savefig('Problem3.pdf')` (make sure that you do this before or instead of `plt.show()`, otherwise 
#you will save a blank image). To upload the pdf to your github, open the git console and run `git add Problem3.pdf` before 
#committing and pushing. This problem will be graded manually.

from pylab import *

figure()

in_val = 30
xx1 = linspace(x[0], x[-1], 50)
xx2 = linspace(x[-1], in_val, 50)

plot(x, y, 'go')
plot(xx1, estimate_value(xx1), 'r-')
plot(xx2, estimate_value(xx2), 'r:')
plot((in_val), estimate_value(in_val), 'bo', markersize=12)
xlabel('x')
ylabel('y')

savefig('Problem3.pdf')
show()
#%%
#1.Write a function `get_average_distance_unit_square(n)` that estimates the average distance
   #between two random points (uniformly drawn) in the unit square
  # based on $n$ different random (uniform) draws of pairs of points in the unit square.
  
import numpy as np
from numpy import random
from numpy.linalg import norm

def get_average_distance_unit_square(n):
    s = random.rand(n,2)
    d = s[1:]-s[:-1]
    return np.mean(norm(d,axis=1))
#%%
#2. Write a function `get_average_distance_unit_disk(n)` that estimates the average distance
   #between two random points (uniformly drawn) in the unit disk
   #based on $n$ different random (uniform) draws of pairs of points in the unit disk.
   #If you intend to generate the random points via random polar coordinates,
   #be very careful with how you use the random radial component when generating the points.
   #Recall that the probability of a point landing in a particular subset of the disk should be proportional to that subset's area.

import numpy as np
from numpy import random
from numpy.linalg import norm

def get_average_distance_unit_disk(n):
    
    N = 2*int(n*4/np.pi)
    s = random.rand(N, 2)*2 - 1
    s = s[norm(s, axis=1) <= 1]
    d = s[1:] - s[:-1]
    distances = norm(d, axis=1)
    return np.mean(distances)
#%%
#3. Write two functions: `one_dimensional_random_walk(steps, up2down_chance_ratio)`
   #and `compare_random_walks(n, reference_pos, steps, up2down1, up2down2)`.
   #The first should simulate a simple one-dimensional random walk with `steps` many iterations
   #where at each iteration the chance to go up divided by the chance to go down
   #(i.e. the probability of adding $1$ to your previous position
   #divided by the probability of subtracting $1$ from the previous position)
   #is the input `up2down_chance_ratio`.
   #This function should return the ending position of the random walk.
   #The second function should compare how likely two different kinds of random walks
   #are to stick close to a desired ending position `reference_pos`.
   #The second function should call the first $n$ times with parameters `steps` and `up2down1`
   #and again $n$ times with parameters `steps` and `up2down2`.
   #For each of the $n$ instances, the function is supposed to decide which of the $2$ random walks was closer to `reference_pos`.
   #The second function should output two counters, representing how many times the first type of random walk
   #got closer to `reference_pos` and, respectively, how many times the second type of random walk ended up closer.
   #To get an intuition of what this means, observe what happens when `steps` is $10000$,
   #`reference_pos` is $100$, the first random walk is uniform (i.e., the ratio `up2down1` is $1$)
   #and the second random walk is slightly biased towards going up (e.g. `up2down2` is $1.1$).
   #Think about the expected value of the ending position for each of these two types of random walks.

import numpy as np
from numpy import random

def one_dimensional_random_walk(steps, up2down_chance_ratio):
    # Probability of going up
    p = up2down_chance_ratio/(1+up2down_chance_ratio)
    return 2*sum(random.rand(steps)<p) - steps

def compare_random_walks(n, reference_pos, steps, up2down1, up2down2):
    # your code here
    walk1 = np.array([one_dimensional_random_walk(steps, up2down1)
                      for i in range(n)])
    walk2 = np.array([one_dimensional_random_walk(steps, up2down2)
                      for i in range(n)])
    counter1 = sum(abs(walk1-reference_pos)<abs(walk2-reference_pos))
    counter2 = sum(abs(walk2-reference_pos)<abs(walk1-reference_pos))
    return counter1, counter2

#Using the functions
n = 100
reference_pos = 10
steps = 50
up2down1 = 1    # Equal chance of up or down
up2down2 = 1.1  # Slightly more chance of moving up

# Compare the walks
result = compare_random_walks(n, reference_pos, steps, up2down1, up2down2)
print(f"Type 1 closer: {result[0]} times, Type 2 closer: {result[1]} times")
#%%
#4. Write a function `get_random_subset_of_naturals_up_to_20()` that outputs a random subset of the set of integers
   #between $1$ and $20$ in the format of a `numpy` array.
   #The draw of the subset should be uniform, i.e., any subset should in principle have the same chance to be outputted by your function.
   #This problem will be graded manually.
   #For $2$ out of the $5$ points allotted to this problem, you can write your function however you wish.
   #To get $5$ points, your function is allowed to make a single call to the `numpy.random.randint()`
   #method but it cannot make use of any other random methods.

def get_random_subset_of_naturals_up_to_20():
    randomnumber = np.array([random.randint(0,2**20)],dtype='>i4')
    mask = np.unpackbits(randomnumber.view(np.uint8))
    boolean_mask = np.array(mask[-20:],dtype=bool)
    return np.arange(1,21)[boolean_mask]
#%%
#1. Use sympy to solve the system of equations $x^2+y^2=r^2$, $2 y=4 x+1$.
#Your code should define a list of dictionaries named `sol` which contains replacement expressions for `x` and `y` in terms 
#of the symbolic parameter `r`.

import sympy as sp

x, y, r = sp.symbols('x y r')
eq1 = x**2 + y**2 - r**2
eq2 = 2*y - 4*x - 1

sol = sp.solve([eq1,eq2], x, y, dict=True)
#%%
#2. Plot the two curves described by the equations from Problem 1 for $r=2$ into a single coordinate system.  
#Further, plot the solution points derived in Problem 1 as two visible dots into the same coordinate axes.
#This question will be graded manually based on the graphical output.  Be sure to label your coordinate axes and choose a 
#sensible coordinate range for plotting.  Before committing your submission, issue `git add Problem2.pdf` to make sure that 
#your graph will be uploaded to Github.

from pylab import *

figure()

x = linspace(-2.5, 2.5, 200)
y = x

xx, yy = meshgrid(x, y)

contour(xx, yy, xx**2 + yy**2 - 4, [0])
contour(xx, yy, 2*yy - 4*xx - 1, [0])

# This will force an aspect ratio of 1 so that the circle looks round...
gca().set_aspect('equal')

# Now plot the solutions from Problem 1:

x = [x.subs(s).subs(r,2) for s in sol]
y = [y.subs(s).subs(r,2) for s in sol]

plot(x, y, 'o')

xlabel('x')
ylabel('y')

# The following command will save your plot as a PDF file:
savefig("Problem2.pdf")
show()
#%%
#1. Suppose you have a dataset representing the tasks completed by 100 employees in two different departments, A and B. 
#Each department comprises 50 employees. The 'tasks_completed' column contains random integers between 10 and 25 (inclusive), 
#indicating the number of tasks completed by each employee. Based on the assumption, create a DataFrame and determine whether there 
#is a statistically significant difference in the average number of tasks completed between Department A and Department B. 
#Conduct an independent samples t-test to compare the average tasks completed by employees in Department A and Department B. 
#Interpret the result considering a significance level (alpha) of 0.05, and provide a box plot to visualize the distributions.
#This question will be graded manually. Before committing your submission, issue git add Problem1.pdf to make sure that your plot 
#will be uploaded to Github.

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create the dataset
data = {
    'employee': range(1, 101),
    'department': ['A', 'B'] * 50,
    'tasks_completed': np.random.randint(10, 25, 100)
}

df = pd.DataFrame(data)

# Visualize the distributions
plt.figure(figsize=(10, 6))
sns.boxplot(x='department', y='tasks_completed', data=df)
plt.title('Tasks Completed by Department')
plt.xlabel('Department')
plt.ylabel('Tasks Completed')
plt.savefig("Problem1.pdf")
plt.show()

# Conduct independent samples t-test
t_stat, p_value = stats.ttest_ind(df['tasks_completed'][df['department'] == 'A'],
                                  df['tasks_completed'][df['department'] == 'B'])

# Interpret the results mathematically
alpha = 0.05
if p_value < alpha:
    print(f"Reject the null hypothesis; there is a significant difference in mean scores (p-value={p_value:.4f}).")
else:
    print(f"Fail to reject the null hypothesis; no significant difference in mean scores (p-value={p_value:.4f}).")
#%%
#2. It is known that any rational function of $\sqrt{2}$ with rational coefficients can be written in the canonical representation 
#$r(\sqrt{2}) = a + b \sqrt{2}$, where $a$ and $b$ are again rational numbers.Write a function `canonical_representation(r)` that 
#takes as argument the rational function $r$ as a sympy function, and returns the coefficients `(a,b)` as a Python tuple.

import sympy as sp

q = sp.sqrt(2)

def canonical_representation(r):
    
    s = sp.simplify(r(q))
    a = s.subs(q, 0)
    b = s.coeff(q)
    return (a, b)
#%%
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html
#https://ethanweed.github.io/pythonbook/03.01-descriptives.html



















