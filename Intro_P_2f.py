#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:31:25 2023

@author: maryamramezaniziarani
"""

#%%
# for and while loop
#%%
#Exercise1:Write a Python program to find the largest number in a given list of numbers
#%%
arr = [2, 4, 6, 8]
max_num = arr[0]

for num in arr:
    if num > max_num:
        max_num = num
print(max_num)
#%%
i = 1
while i < 11:
    print(i)
    i += 1
#%%
#Exercise2:print out the numbers from 100 to 0 in steps of 10, using while loop
#%%
i = 100
while i >= 0:
    print(i)
    i -= 10
#%%
## Def function
def add_numbers(num1, num2):
    sum = num1 + num2
    return sum
print(add_numbers(3, 5))
#%%
#Exercise3:What is the area of a rectangle with a length of 2 and a width of 4? (use def function)
#%%
def area_of_rectangle(length, width):
    area = length * width
    return area
print(area_of_rectangle(2, 4))
#%%
#Euclidean_algorithm
#%%
#Exercise4:Write a Python function, `euclidean_algorithm(a, b)`, that implements
#the Euclidean algorithm to find the greatest common divisor (GCD) of 
#two positive integers 'a' and 'b'. The function should return the GCD.
#%%
def euclidean_algorithm(a, b):
    while b:
        a, b = b, a % b
    return a

# Example of finding the GCD of 48 and 18
result = euclidean_algorithm(48, 18)
print("GCD:", result)  # Output: GCD: 6
#%%
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
b = np.arange(10).reshape(2, 5) 
print(b)
#%%
type(b)
#%%
b.shape
#%%
b.size
#%%
b.ndim
#%%
#Exercise5:Create a 3-dimensional array using np.arange() and
#reshape and check for shape, size, dimension
#%%
c = np.arange(1, 19).reshape((3, 3, 2))
print(c)
#%%
c.shape
#%%
c.size
#%%
c.ndim
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)





