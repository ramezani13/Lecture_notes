#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: maryamramezaniziarani
"""

# SymPy; symbolic mathematics
#%%
import sympy as sp

#%% manipulate numbers exactly using SymPy
#square root of a number that is a perfect square
import math
math.sqrt(4)
#%% square root of a number that isn’t a perfect square
math.sqrt(8)
#%% square root of a number that isn’t a perfect square
sp.sqrt(5)
#%% square root of a number that isn’t a perfect square; symbolically simplified
sp.sqrt(8)
#%% Exercise1:
#Calculate the square root of a perfect square, say 25, using both the math module and SymPy.
#Calculate the square root of a non-perfect square, say 10, using both the math module and SymPy.
#Calculate the square root of a non-perfect square, say 18, using SymPy, symbolically simplified result.

# 1. Square root of a perfect square
perfect_square_math = math.sqrt(25)
print(perfect_square_math)

perfect_square_sympy = sp.sqrt(25)
print(perfect_square_sympy)

# 2. Square root of a non-perfect square
non_perfect_square_math = math.sqrt(10)
print(non_perfect_square_math)

non_perfect_square_sympy = sp.sqrt(10)
print(non_perfect_square_sympy)


# 3. Square root of a non-perfect square, symbolically simplified
non_perfect_square_symbolic = sp.sqrt(18)
print(non_perfect_square_symbolic)

#%%
#Symbolic Variables

x, y, z = sp.symbols('t y z')

# Print the symbolic variables
print("x:", x)
print("y:", y)
print("z:", z)
#%%
#Symbolic Expressions

expr = x**2 + 2*x + 1

# Print the expression
print("Expression:", expr)
#%%
from sympy import symbols

x= symbols('a')
expr = x**2 + 2*x + 1

print("Expression:", expr)
#%%
expr = expr + 1
print("Expression:", expr)
#%%
expr = x*expr
print("Expression:", expr)
#%%
from sympy import expand, factor
expanded_expr = expand(expr)
print("Expanded Expression:", expanded_expr)
#%%
factor_expr = factor(expanded_expr)
print("Factor Expression:", factor_expr)
#%%
#Exercise2: Consider the expression expr = 2*t**3 - 6*t**2 + 3*t, a. Expand the Expression, b. Factor the Expanded 
#Expression.

from sympy import symbols, expand, factor

# Define symbolic variable
t = symbols('t')

# Given expression
expr = 2*t**3 - 6*t**2 + 3*t

# a. Expand the Expression
expanded_expr = expand(expr)
print("Expanded Expression:", expanded_expr)

# b. Factor the Expanded Expression
factored_expr = factor(expanded_expr)
print("Factored Expression:", factored_expr)

#%%
#Solving a System of Equations with SymP

# x + y = 5
# 2*x - y = 1

from sympy import symbols, Eq, solve

# Define Symbolic Variables
x, y = symbols('x y')

# Set Up Equations
eq1 = Eq(x + y, 5)
eq2 = Eq(2*x - y, 1)

# Solve the System
solution = solve([eq1, eq2], [x, y])
solution

#%%
# Exercise3: Consider the system of equations:

#3x+2y−z=4
#2x−y+3z=−6
#x+3y−2z=8

#Use SymPy to solve the system of equations.

#Determine the type of solution the system has (unique solution, no solution, or infinitely many solutions).

#Extract and print the solutions, including any symbolic expressions if present.


from sympy import symbols, Eq, solve

# Define Symbolic Variables
x, y, z = symbols('x y z')

# Set Up Equations
eq1 = Eq(3*x + 2*y - z, 4)
eq2 = Eq(2*x - y + 3*z, -6)
eq3 = Eq(x + 3*y - 2*z, 8)

# Solve the System
solution = solve([eq1, eq2, eq3], [x, y, z])
print(solution)

# Check the type of solution
if solution:
    if all(isinstance(sol, int) or isinstance(sol, float) for sol in solution.values()): #method that returns a view of all values in the dictionary
        solution_type = "Unique solution"
    else:
        solution_type = "Infinitely many solutions (symbolic)"
else:
    solution_type = "No solution"

# Extract and Print Solutions
print(f"Solution Type: {solution_type}")

if solution_type == "Unique solution":
    x_val = solution[x] #access the value associated with the key x in the solution dictionary.
    y_val = solution[y]
    z_val = solution[z]
    print(f"Unique Solution: x = {x_val}, y = {y_val}, z = {z_val}")
elif solution_type == "Infinitely many solutions (symbolic)":
    print("Symbolic solutions present. Displaying in symbolic form:")
    for sol_key, sol_val in solution.items(): #displays a list of a dictionary's key-value tuple pairs
        print(f"{sol_key} = {sol_val}")
else:
    print("No solution")

#%%

# Exercise4: Consider the system of equations:

#2x+y=5
#3x-2y=8

#Use SymPy to solve the system of equations.

#Determine the type of solution the system has (unique solution, no solution, or infinitely many solutions).

#Extract and print the solutions, including any symbolic expressions if present.

#Plot the Equations and Solution Point

import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Define Symbolic Variables
x, y = sp.symbols('x y')

# Step 2: Set Up Equations
eq1 = sp.Eq(2*x + y, 5)
eq2 = sp.Eq(3*x - 2*y, 8)

# Step 3: Solve the System
solution = sp.solve([eq1, eq2], [x, y])
print(solution)

# Step 4: Determine the Type of Solution
if solution:
    if all(isinstance(sol, int) or isinstance(sol, float) for sol in solution.values()):
        solution_type = "Unique solution"
    else:
        solution_type = "Infinitely many solutions (symbolic)"
else:
    solution_type = "No solution"

# Step 5: Extract and Print Solutions
print(f"Solution Type: {solution_type}")

if solution_type == "Unique solution":
    x_val = solution[x]
    y_val = solution[y]
    print(f"Unique Solution: x = {x_val}, y = {y_val}")

    # Step 6: Plot the Equations and Solution Point
    x_vals = np.linspace(-5, 5, 100)
    y_vals_eq1 = 5 - 2*x_vals  # Rearrange eq1 for y
    y_vals_eq2 = (3*x_vals - 8) / 2  # Rearrange eq2 for y

    plt.plot(x_vals, y_vals_eq1, label='2x + y = 5')
    plt.plot(x_vals, y_vals_eq2, label='3x - 2y = 8')

    plt.scatter(float(x_val), float(y_val), color='red', label='Solution Point')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('System of Equations')
    plt.legend()
    plt.grid(True)
    plt.show()

elif solution_type == "Infinitely many solutions (symbolic)":
    print("Symbolic solutions present. Unable to plot.")
else:
    print("No solution. Unable to plot.")
#%%
from sympy import symbols, Eq, solve

x, y, z = symbols('x y z')

eq1 = Eq(x + y + z, 1)
eq2 = Eq(2*x + 2*y + 2*z, 2)

solution = solve([eq1, eq2], [x, y, z])
print(solution)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html
