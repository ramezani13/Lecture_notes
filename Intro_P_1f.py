#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:27:29 2023

@author: MRZ
"""

##Variables and data types
#%%
# Numeric int, float
#%%
a=1
type(a)
#%%
b= 2.8
type(2.8)
#%%
# Sequence list, tuple
#%%
type([4, 6, 7])
#%%
type((1, 5))
#%%
# String str
#%%
type("I like Math!")
#%%
# Mapping dict
#%%
type({"number of student": 8, "p": 5})
#%%
#set
#%%
type({"number of student", "p"})
#%%
#Boolean 
#%%
print(type(1 == 2))
#%%
c=("I like Math!")
print(c[2])
#%%
#Excercise1: Assign a list to a variable and check the type and print an element using index!
#%%
b=[4,5,8]
type(b)
print(b[2])
#%%
#Excercise2: Assign a float to a variable and use the print() function
#to print out the following sentence "the type of 2.8 is <class 'float'>"
#%%
b= 2.8
print ("the type of", b ,"is", type(b))
#%%
# Identity and Equality
#%%

a = [1, 2, 3]
b = a               # b is a reference to the same list as a
c = [1, 2, 3]       # c is a different list with the same values as a

identity_check = print(a is b)    # True, they reference the same object
equality_check = print(a == c)    # True, they have the same values

#%%
# Mutable (list, dict, set) vs. Immutable (string, numbers, tuple)
#%%
# Lists are mutable
my_list = [1, 2, 3]
my_list[0] = 4
print(my_list)  # Output: [4, 2, 3]

# Strings are immutable
my_string = "Hello"
# my_string[0] = 'h'  # This will raise an error
#%%
#Excercise3: Check this for (dict, set, tuple)
#%%
# Dictionaries are mutable
my_dict = {'name': 'Alice', 'age': 25}
my_dict['age'] = 26
print(my_dict)  

# Sets are mutable
my_set = {1, 2, 3}
my_set.add(4)
print(my_set)  

# Tuples are immutable
my_tuple = (1, 2, 3)
my_tuple[0] = 4  
print(my_tuple)

#%%    
# Arithmetic operators
#%%
a = 5
b = 2
addition = a + b
print(a+b)
subtraction = a - b
print(a-b)
multiplication = a * b
print(a*b)
division = a / b
print(a/b)
remainder = a % b
print(a%b)
exponentiation = a ** b
print(a**b)
print(1+5*4)
print((1+5)*4)

#%%
# Comparison operators
#%%
x = 5
y = 10
equals = x == y
not_equals = x != y
greater_than = x > y
less_than = x < y
print(x > y)
#%%
#Conditional Statements
#%%
grade=60

if grade>50:
    print("you pass the test")
#%%
a=34
if a<20:
    print("you are a winner")
else:
    print("you are a loser")
#%% #IndentationError
grade=60

if grade>50:
print("you pass the test")
#%%
#Excercise4: Given two numbers a and b, write a Python code
#to check if a is divisible by b.
#%%
a=10
b=5
if (a % b == 0): 
    print("x is divisible by y") 
else: 
    print("x is not divisible by y")
#%%
# for and while loops
#%%
animals = ["dog", "cat", "bird"]
for x in animals:
    print(x)
#%%
animals = ["dog", "cat", "bird"]
for x in animals:
    print(x)
    if x == "cat":
        break
#%%
x = range(10, 20)
for n in x:
    print(n)
#%%
x = range(10, 20,2)
for n in x:
    print(n)
#%%
#Exercise5: What numbers between 2 and 30 are even and odd?
#%%
for x in range(2, 30):  
    if x % 2 == 0: 
        print(x, "is a Even Number")
    else: 
        print(x, "is a Odd Number")
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#Prof. Marcel Oliver (Introduction to Programming, Winter Semester 2022/2023)
