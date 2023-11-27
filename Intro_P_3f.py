#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:13:47 2023

@author: maryamramezaniziarani
"""
#%%
#Array and numpy
#%%
import numpy as np
y = np.array([[4, 5, 0], [3, 7, 9], [2, 1, 2]])
print(y)
#%%
z = np.array([[[1,2,3], [4,5,6], [7,8,9]], [[10,11,12], [13,14,15], [16,17,18]]])
print(z)
#%%
#Exercise1:Extract the value 15 from the z array above by indexing
#%%
z[1, 1, 2]
#%%
##array slice
#%%
z[1, 2, :]
#%%
y[:2, :]
#%%
y[1:, :]
#%%
z[::-1]
#%%
x="Maryam"
x[::-1]

#%%
#Exercise2:Extract the 2nd layer, first and 2nd rows and all
#columns from the z array above by slice and indexing
#%%
z[1, 0:2, :]
#%%floating point and machine epsilon

#%%
#Exercise3:
#%Write a Python function sqrt_apx(x, epsilon) that approximates the 
#square root of a positive number x with a given epsilon. 
#The function should use an iterative approach to find an approximation of the
#square root such that the difference between the square of the approximation 
#and x is less than or equal to the specified epsilon

def sqr_apx(x, epsilon):
    
    if x < 0:
        raise ValueError("Input value can't be negative.")

    
    guess = 1.0

    
    while abs(guess * guess - x) > epsilon:
        # Update the guess using the formula: new_guess = 0.5 * (old_guess + x / old_guess)
        guess = 0.5 * (guess + x / guess)

    return guess

# Example:
x = 16.0
epsilon = 1e-6  # Tolerance level

approximated_result = sqr_apx(x, epsilon)
print(f"Approximated square root of {x} is approximately {approximated_result}")
#%%
#Newton-Raphson formula (root-finding algorithm) for finding the square root 
f(x) = guess^2 - x

The derivative of f(x), denoted as f'(x), is f'(x) = 2 * guess. We use the Newton-Raphson formula:

x1 = x0 - f(x0) / f'(x0)

Substituting f(x) and f'(x), we get:

x1 = x0 - (x0^2 - x) / (2 * x0)

Simplifying:

x1 = x0 - (x0^2 - x) / (2 * x0)
x1 = x0 - (x0^2 / (2 * x0)) + (x / (2 * x0))

We can further simplify the first term:

x0^2 / (2 * x0) = (x0 * x0) / (2 * x0) = x0 / 2

So, the formula becomes:

x1 = x0 - (x0 / 2) + (x / (2 * x0))

Now, let's factor out 1/2 from both terms:

x1 = 0.5 * (2 * x0 - x0 + x / (x0))

Further simplifying the terms inside the parentheses:

x1 = 0.5 * (x0 + x / (x0))

This is equivalent to:

new_guess = 0.5 * (old_guess + x / old_guess)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#Prof. Marcel Oliver (Introduction to Programming, Winter Semester 2022/2023)
















