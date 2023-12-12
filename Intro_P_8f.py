#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
##reading the data from the file, saving the data to the file
#%%
b = np.genfromtxt('sample.txt', delimiter=';')
print(b) 
#%%
b = np.genfromtxt('sample.txt', delimiter=';', skip_header=1)
print(b)
#%%
np.savetxt("y_test.txt", y, delimiter=",", header="name,id,region")
#%%
#Exercise1: Create a 2D array and use the np.savetxt() function to save the data array to a text 
#file named "mydata.txt" with a comma delimiter and a header line Col1,Col2,Col3,Col4

data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

np.savetxt("mydata.txt", data, delimiter=",", header="Col1,Col2,Col3,Col4", fmt="%.3e")

print("Data saved to mydata.txt file.")
#%%
## ploting an array
#%%
import matplotlib.pyplot as plt

#%%
plt.plot(y[1, :]);
#%%
plt.plot(y[:, 1]);
#%%
plt.plot(y);
#%%
plt.plot(y, color='red');
#%%
plt.plot(y[:, 0], color='red', label='red')
plt.plot(y[:, 1], color='green', label='green')
plt.plot(y[:, 2], color='blue', label='blue')
plt.legend();
plt.title('Line Plot for Each Column in y')
plt.xlabel('X-axis (Assuming Index)')
plt.ylabel('Y-axis (Values in Each Column)')
plt.show()
#%%
plt.imshow(y);
plt.colorbar()
#%%
#Exercise2: Given an 2D array of numbers, plot a graph that displays the sum of 
#each element in the column.

arr = [[1,2,3],
       [4,5,6],
       [7,8,9]]
column_sums= np.sum(arr, axis=0)
print(column_sums)
plt.plot(column_sums)
plt.title('Sum of elements in each column')
plt.xlabel('column')
plt.ylabel('Sum of elements')
plt.show()
#%%
#Histograms, the distribution of a single variable
y = np.array([[4, 5, 0], [3, 7, 9], [2, 1, 2]])
h = y.flatten()
print(h)
plt.hist(h);
#%%
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

plt.hist(data, bins=[0,2,4,6,8,10])
plt.title('My Histogram')
plt.xlabel('Interval')
plt.ylabel('Frequency')
plt.show()
#%%
#Line Plots, relationships between variables

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

plt.plot(x, y)
plt.show()

#%%
#Linear Regression Fit and trend line

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# Fit a linear regression line
slope, intercept = np.polyfit(x, y, 1)
trend_line = np.polyval([slope, intercept], x)

# Plot the original data points
plt.plot(x, y, 'o', label='Data Points')

# Plot the trend line
plt.plot(x, trend_line, label='Trend Line', color='red')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Trend Line')
plt.legend()

# Show the plot
plt.show()
#%%
#Scatter Plots, the distribution of individual data points

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

plt.scatter(x, y)
plt.show()
#%%
#Bar Plots, categorical data

categories = ['A', 'B', 'C']
values = [4, 7, 2]

plt.bar(categories, values)
plt.show()

#%%
#Exercise3:
#Consider a scenario where the relationship between two variables, `x` and `y`, is described by a quadratic 
#function: y = ax^2 + bx + c

#a).  Write a Python function, `generate_quadratic_data(a, b, c, num_points)`, that takes coefficients a, b, and
#c, and the number of data points `num_points`, and generates synthetic data according to the quadratic function.
#Assume x values in the range from -10 to 10, using x = np.linspace(-10, 10, num_points).
    
#b). Use the function to generate a dataset with a=1, b=-2, c=1, and num_points=100


#c). Write another function, plot_quadratic_scatter(x, y), that takes the generated x and y values and creates a 
#scatter plot.


def generate_quadratic_data(a, b, c, num_points):
    x = np.linspace(-10, 10, num_points)
    y = a * x**2 + b * x + c
    return x, y

def plot_quadratic_scatter(x, y):
    plt.scatter(x, y, label='Generated Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quadratic Data Scatter Plot')
    plt.legend()
    plt.show()

# Generate quadratic data
a, b, c = 1, -2, 1
num_points = 100
x_data, y_data = generate_quadratic_data(a, b, c, num_points)

# Plot the scatter plot
plot_quadratic_scatter(x_data, y_data)

#%%
#Exercise4:Create a Python script using NumPy and Matplotlib to fit a quadratic curve to the given data points
#and plot the results. The data points are:
    
# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 6, 9, 15])

# Fit a quadratic curve to the data
coefficients = np.polyfit(x, y, 2)
print(coefficients)

# x values for the curve
x_curve = x

# Use polyval to calculate y values for the curve
y_curve = np.polyval(coefficients, x_curve)
print(y_curve)

# Plot data points as blue dots
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the quadratic curve as a red curve
plt.plot(x_curve, y_curve, color='red', label='Quadratic Trendline')

# Set axis labels
plt.xlabel('X')
plt.ylabel('Y')

# Set the legend
plt.legend()

# Show the plot
plt.show()
#%% 

#OR:
    
# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 6, 9, 15])

# Fit a quadratic curve to the data
coefficients = np.polyfit(x, y, 2)
quadratic_curve = np.poly1d(coefficients)
print(quadratic_curve)

# Generate x values for the curve
x_curve = x

# Plot data points as blue dots
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the quadratic curve as a red curve
plt.plot(x_curve, quadratic_curve(x_curve), color='red', label='Quadratic Trendline')

# Set axis labels
plt.xlabel('X')
plt.ylabel('Y')

# Set the legend
plt.legend()

# Show the plot
plt.show()

#%%
#Exercise5: Consider a power-law function (y = cx^alpha), where (c = 2) and (alpha = 1.5). Create a plot of 
#this power-law function for (x) values in the range ([1, 10]), # Given parameters c = 2 and alpha = 1.5.
#Use NumPy and Matplotlib to generate the plot.

# Define the power-law function
def power_law(x, c, alpha):
    return c * x ** alpha

# Given parameters
c = 2
alpha = 1.5

# Generate x values in the range [1, 10]
x_values = np.linspace(1, 10, 20)
print(x_values)

# Calculate corresponding y values using the power-law function
y_values = power_law(x_values, c, alpha)
print(y_values)

# Create a plot
plt.plot(x_values, y_values, label='Power Law')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Power Law Plot')
plt.legend()
plt.grid(True)
plt.show()

#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)





