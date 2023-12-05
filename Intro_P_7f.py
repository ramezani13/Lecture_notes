#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:36:01 2023

@author: maryamramezaniziarani
"""

#Array and numpy
#%%
import numpy as np
x = np.array([5, 4, -7, 11])
print(x)
#%%
y = np.array([[4, 5, 0], [3, 7, 9], [2, 1, 2]])
print(y)
#%%
z = np.array([[[1,2,3], [4,5,6], [7,8,9]], [[10,11,12], [13,14,15], [16,17,18]]])
print(z)
#%%
#Boolean Indexing and Logical Operations 
#%%
x<5 
#%%
x==11
#%%
~(x > 1)
#%%
#Masks for subsetting
#%%
x[x < 5]
#%%
x[x < 5]=40
x
#%%
#Exercise1: Substitute the values from array x which are not greater than 20,or equal to 11 with 30
#(use logical operator not and or)

x[~(x > 20)| (x==11)]=30
x
#%%
##Vectorizing
#%%
x+2
#%%
x**2
#%%
np.abs(x)
#%%
np.max(x)
#%%
np.median(x)
#%%
#Exercise2: a) Write a Python program to find the median of a given array of numbers with for and if 
#loop. b) Write a Python program to find the median of a given array of numbers using vectorizing

m=[2,3,5,7,8,9,11]

#using for loop

array_len = len(m)

if array_len % 2 == 0:
    median = (m[int(array_len/2)] + m[int(array_len/2 -1)])/2
else:
    median = m[int(array_len/2)]

print(median)

# b) np.median(m)
#%%
np.sum(x)
#%%
np.sum(x<35)
#%%
np.sum(y, axis=0)
#%%
np.sum(y, axis=1)
#%%
##row-column
np.sum(z, axis=0)
#%%
##column-layer
np.sum(z, axis=1)
#%%
##row-layer
np.sum(z, axis=2)
#%%
#Exercise3: Calculate the mean of all column in each layer from array z
np.mean(z,axis=1)
#%%
#Exercise4: Consider the following NumPy array representing a matrix:

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

#a) Write a Python program to find the sum of the diagonal elements of a given 
#square matrix without using the np.trace function and without vectorizing. Use 
#loops for iteration.

#b) Implement a function that takes a square matrix as input and returns True if 
#the matrix is symmetric and False otherwise. Avoid using vectorized NumPy 
#operations and use loops for element comparison.

#c) Write a program to calculate the product of each column in the matrix without 
#using np.prod and without vectorizing. Use nested loops for the calculation.

#a)

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

# Solution for part (a) without vectorizing
diagonal_sum = 0
for i in range(len(matrix)):
    diagonal_sum += matrix[i, i]

print("Sum of diagonal elements without vectorizing:", diagonal_sum)

#b)

def is_symmetric(matrix):
    # Solution for part (b) without vectorizing
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != matrix[j, i]:
                return False
    return True

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

symmetric_result = is_symmetric(matrix)
print("Is the matrix symmetric without vectorizing?", symmetric_result)


#c) 

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

# Solution for part (c) without vectorizing
rows, cols = matrix.shape
column_products = np.ones(cols)

for j in range(cols):
    for i in range(rows):
        column_products[j] *= matrix[i, j]

print("Product of each column without vectorizing:", column_products)

#%%
# a)

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

# Solution for part (a)
diagonal_sum = np.trace(matrix)
print("Sum of diagonal elements:", diagonal_sum)

# b)


def is_symmetric(matrix):
    # Solution for part (b)
    return np.allclose(matrix, matrix.T)

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

symmetric_result = is_symmetric(matrix)
print("Is the matrix symmetric?", symmetric_result)

# c)


matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

# Solution for part (c)
column_products = np.prod(matrix, axis=0)
print("Product of each column:", column_products)
   
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)

                 







