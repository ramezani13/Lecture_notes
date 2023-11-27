#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:37:39 2023

@author: maryamramezaniziarani
"""
#%%
#Scope 
#%%
# global scope variable
global_number = 42

def example_function():
    # local scope variable
    local_number = 7
    print("Inside the function:")
    print("Global variable:", global_number)  # global variable within the function
    print("Local variable:", local_number)    # local variable within the function

example_function()

print("Outside the function:")
print("Global variable:", global_number)
#print("Local variable:", local_number)
#%%

def outer_function():
    outer_variable = "I am outer"

    def inner_function():
        nonlocal outer_variable
        inner_variable = "I am inner"
        
        # Accessing and modifying the outer_variable from the enclosing scope
        print("Inside inner function:")
        print("Before modification - outer_variable:", outer_variable)
        outer_variable = "Modified outer from inner"
        print("After modification - outer_variable:", outer_variable)
        print("inner_variable:", inner_variable)

    inner_function()

    # Accessing outer_variable outside inner_function
    print("Outside inner function - outer_variable:", outer_variable)

# Call the outer function
outer_function()
#%%
#Exercise1:
def outer_function(x):
    outer_variable = x

    def inner_function(y):
        # Challenge 1: Use nonlocal to modify outer_variable
        inner_variable = y

        # Challenge 2: Try to access x from outer_function inside inner_function
        
    inner_function(20)

    # Challenge 3: Try to access inner_variable outside inner_function
    

# Challenge 4: Access outer_variable outside outer_function

outer_function(10)
#%%
def outer_function(x):
    outer_variable = x

    def inner_function(y):
        nonlocal outer_variable  # Challenge 1: Use nonlocal to modify outer_variable
        inner_variable = y

        # Challenge 2: Try to access x from outer_function inside inner_function
        print("Accessing x from outer_function inside inner_function:", x)
        print("Inside inner function:")
        print("outer_variable:", outer_variable)
        print("inner_variable:", inner_variable)

    inner_function(20)

    # Challenge 3: Try to access inner_variable outside inner_function
    #print("Trying to access inner_variable outside inner function:", inner_variable)

# Challenge 4: Access outer_variable outside outer_function
#print("Accessing outer_variable outside outer function:", outer_variable)

# Call the outer function with an argument
outer_function(10)
#%%
#Functions as First-Class Objects
#%%
#Assigning Functions to Variables:
    
def square(x):
    return x * x

def cube(x):
    return x * x * x

function_variable = square
print(function_variable(5))

#Passing Functions as Arguments:
    
def apply_operation(func, y):
    return func(y)

result = apply_operation(cube, 3)
print(result)  

#Returning Functions from a Function:

def get_function(power):
    if power == 2:
        return square
    else:
        return cube

power_of_two = get_function(2)
print(power_of_two(4))  

#%%
#Exercise2:
    
#Write a Python function called manipulate_numbers that takes two numbers, x and y, 
#and a function manipulation_func as arguments. The function should apply the provided 
#manipulation_func to the two numbers and return the result. Additionally, 
#demonstrate the use of this function with various manipulation functions.

#manipulation functions:

#addition, subtraction, multiplication, power

#Provide examples of using the manipulate_numbers function with each manipulation function. 
  

# Manipulation function: addition
def addition(x, y):
    return x + y

# Manipulation function: subtraction
def subtraction(x, y):
    return x - y

# Manipulation function: multiplication
def multiplication(x, y):
    return x * y

# Manipulation function: power
def power(x, y):
    return x ** y

def manipulate_numbers(x, y, manipulation_func):
    return manipulation_func(x, y)

# Using the manipulate_numbers function
result1 = manipulate_numbers(5, 3, addition)          
result2 = manipulate_numbers(7, 4, subtraction)      
result3 = manipulate_numbers(2, 6, multiplication)    
result4 = manipulate_numbers(2, 3, power)             
   
#%%
# Goals of object-oriented programming 

# Define a class named 'Dog'
class Dog:
    def __init__(self, name, age):
        # Constructor method initializes attributes
        self.name = name
        self.age = age

    def bark(self):
        # Method to make the dog bark
        print("Woof! Woof!")

# Create instances (objects) of the 'Dog' class
dog1 = Dog("Buddy", 3)
dog2 = Dog("Charlie", 5)

# Access attributes and call methods
print(f"{dog1.name} is {dog1.age} years old.")
dog1.bark()  

print(f"{dog2.name} is {dog2.age} years old.")
dog2.bark()  
#%%
#Exercise3:
    
#Create a Python class named BankAccount to model a simple bank account. 
#The class should have the following attributes and methods:

#Attributes:

#account_holder (string): the name of the account holder.
#balance (float): the current balance in the account.

#Methods:

#A constructor method to initialize the account with the account holder's name and an initial balance.
#A method to deposit a specified amount into the account.
#A method to withdraw a specified amount from the account.
#A method to retrieve the current balance of the account.

class BankAccount: # Constructor method initializes attributes
    def __init__(self, account_holder, initial_balance):
        self.account_holder = account_holder
        self.balance = initial_balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited ${amount:.2f}. Current Balance: ${self.balance:.2f}")

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew ${amount:.2f}. Current Balance: ${self.balance:.2f}")
        else:
            print("Insufficient funds. Withdrawal canceled.")

    def get_balance(self):
        return self.balance

# Example Usage
my_account = BankAccount("Alice", 1000.0)
my_account.deposit(500.0)
my_account.withdraw(200.0)
my_account.deposit(100.0)

final_balance = my_account.get_balance()
print(f"{my_account.account_holder}'s Final Balance: ${final_balance:.2f}")

#%%
# Hints for the assignment tomorrow:

# Practice our previous lectures that focus on different data types (mostly strings and dictionaries).
# Practice the list comprehension from the past lectures. [expression for item in iterable if condition]
# Practice string methods such as `partition()` and `split()`.

#specified string
txt = "I am at ku today"

x = txt.partition("am")

print(x)

#seprator
txt2 = "hello, I am Alex, I am 20 years old"

y = txt2.split(", ")

print(y)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#Prof. Marcel Oliver (Introduction to Programming, Winter Semester 2022/2023)












