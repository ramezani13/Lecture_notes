#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 07:13:53 2023

@author: maryamramezaniziarani
"""

#%%
#try-except blocks for error handling 
#%%
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
        return None

# Example:
numerator = 10
denominator = 0

result = safe_divide(numerator, denominator)

if result is not None:
    print(f"The result of {numerator} / {denominator} is: {result}")
else:
    print("Error occurred during division.")
#%%
#Exercise1:
    
#Write a Python function called get_element(my_list, index) that takes a list my_list
#and an index index as input. The function should attempt to access the element at the
#specified index in the list using a try-except block. If the index is valid, it should return
#the element; otherwise, it should print an error message and return None.

def get_element(my_list, index):
    try:
        element = my_list[index]
        return element

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example:
my_list = [10, 20, 30, 40, 50]

index_to_access = 7

element = get_element(my_list, index_to_access)

if element is not None:
    print(f"The element at index {index_to_access} is: {element}")
else:
    print(f"Error: Cannot access element at index {index_to_access}.")

#%%
# Iterators and Generators 
#%%
my_list = [1, 2, 3, 4, 5]
my_iterator = iter(my_list)

while True:
    try:
        value = next(my_iterator)
        print(value)
    except StopIteration:
        break
#%%
def my_generator(start, end):
    current = start
    while current < end:
        yield current
        current += 1

# example
gen = my_generator(1, 5)
for num in gen:
    print(num)
 
    
gen = my_generator(1, 5)
# First iteration
value1 = next(gen)
print(value1)  

# Second iteration
value2 = next(gen)
print(value2)  

# Third iteration
value3 = next(gen)
print(value3)  
#%%
#Exercise2:
#Write a Python program that takes a list of words and uses the iter() function
#to create an iterator. Use a while loop and the next() function to iterate over each word in the
#list. For each word, print the word backward.

word_list = ["python", "programming", "challenge", "iterator"]

iterator = iter(word_list)

while True:
    try:
        word = next(iterator)
        reversed_word = word[::-1]
        print(reversed_word)
    except StopIteration:
        break

#%%
#Exercise3:
#Write a Python generator function that generates a sequence of squares of numbers starting from
#a given start value up to a specified end value (exclusive). The generator should square 
#each number in the sequence.

def square_generator(start, end):
    current = start
    while current < end:
        yield current ** 2
        current += 1

# example
gen = square_generator(2, 6)
for square in gen:
    print(square)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
















