#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: maryamramezaniziarani
"""

#Pandas;structured data
#%%
import pandas as pd
import numpy as np
#%%
g = pd.Series(['a', 'b', 'c'])
print(g)
#%%
#Exercise1: create a Series from an array of numbers using numpy and check for the type

t= pd.Series(np.array([3,5,7,8]))
print(t)
type(t)
#%%
h= pd.Series(np.array([1,7,9]))
print(h)
type(h)
#%%
i= pd.Series([7, 8, 9])
dataframe_dict = {'ID': g, 'value': h, 'condition': i}
dataframe = pd.DataFrame(dataframe_dict)
print(dataframe)

# Save DataFrame to CSV
dataframe.to_csv('neww_file.csv', index=False)
#%%
dataframe.columns
#%%
#Exercise2:create a table (dataframe structure) from multiple series (including boolean and numiric) and check 
#the shape

series1 = pd.Series([1, 2, 3])
series2 = pd.Series([4, 5, 6])
series3 = pd.Series([False, True, False])

df_new = pd.DataFrame({'Val1': series1, 'Val2': series2, 'Val3': series3})
print(df_new)
print(df_new.shape)
#%%
#Descriptive statistics
#%%
df = pd.read_csv('ind_inf.csv')
print(df)
#%%
#Measures of Central Tendency; information about the center or average of a set of values
#%%
df['Age'].mean()
#%%
df['Age'].median()
#%%
df['Age'][0:3].mean()
#%%
a = list(df)
a.mean()
#%%
import statistics #module for basic operations
#%%
statistics.mean(df['Age'])
#%%
statistics.median(df['Age'])
#%%
#Measures of Dispersion; the spread or variability of a dataset
#%%
statistics.variance(df['Age'])
#%%
statistics.stdev(df['Age'])
#%%
#Measures of Shape; distributional characteristics of a dataset
#%%
import scipy.stats as stats # statistical functions and tests

data = [1, 2, 2, 3, 3, 3, 4, 4, 5]

# Skewness
skewness = stats.skew(data)

# Kurtosis
kurtosis = stats.kurtosis(data)

print("Skewness:", skewness)
print("Kurtosis:", kurtosis)
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = [1, 2, 2, 3, 3, 3, 4, 4, 5]

# Calculate skewness and kurtosis
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

# Plot the histogram
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=data, kde=True, ax=ax, color='skyblue')

# Set plot title and labels
ax.set_title("Skewness and Kurtosis Visualization")
ax.set_xlabel("Values")
ax.set_ylabel("Frequency")

# Show the plot
plt.show()
#%%
#Exercise3:create a right-skewed dataset (using exponential distribution) and plot its histogram using
#seaborn, and matplotlib

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Create a right-skewed dataset
right_skewed_data = np.random.exponential(size=1000)
print(right_skewed_data)

# Create a DataFrame
df_right_skewed = pd.DataFrame({'Values': right_skewed_data, 'Skew': 'PosSkew'})
print(df_right_skewed)
# Plot the histogram
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df_right_skewed, x='Values', binwidth=0.5, ax=ax, color='green')

# Set plot title and labels
ax.set_title("Right Skew Histogram")
ax.set_xlabel("Values")
ax.set_ylabel("Frequency")

# Show the plot
plt.show()
#%%
#Percentiles and Quartiles: measures of position in a dataset

# Sample data
data = [12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48, 52, 55, 58]

# Convert data to a pandas Series
series = pd.Series(data)

# Calculate percentiles
percentiles = series.quantile([0.25, 0.5, 0.75])

print("25th Percentile (Q1):", percentiles[0.25])
print("50th Percentile (Q2 or Median):", percentiles[0.5])
print("75th Percentile (Q3):", percentiles[0.75])

# Calculate quartiles
quartiles = series.quantile([0, 0.25, 0.5, 0.75, 1])

print("\nMin:", quartiles[0])
print("25th Percentile (Q1):", quartiles[0.25])
print("50th Percentile (Q2 or Median):", quartiles[0.5])
print("75th Percentile (Q3):", quartiles[0.75])
print("Max:", quartiles[1])
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48, 52, 55, 58]

# Convert data to a pandas Series
series = pd.Series(data)

# Create a boxplot
fig, ax = plt.subplots()
sns.boxplot(x=series, ax=ax)

# Set plot title and labels
ax.set_title("Boxplot with Quartiles")
ax.set_xlabel("Values")

# Show the plot
plt.show()
#%%
#Exercise4:Consider a dataset with 1000 random observations. The data follows a standard normal distribution, 
#and you are asked to find and plot the following:

#The 10th, 25th, 50th (median), 75th, and 90th percentiles.
#The first, second, and third quartiles.
#A box plot representing the quartiles.

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Generate a dataset with 1000 observations
np.random
complex_data = np.random.randn(1000)

# Create a DataFrame
df_complex = pd.DataFrame({'Values': complex_data})
print (df_complex)

# Calculate percentiles
percentiles = df_complex['Values'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

print("10th Percentile:", percentiles[0.1])
print("25th Percentile:", percentiles[0.25])
print("50th Percentile (Median):", percentiles[0.5])
print("75th Percentile:", percentiles[0.75])
print("90th Percentile:", percentiles[0.9])

# Calculate quartiles
quartiles = df_complex['Values'].quantile([0, 0.25, 0.5, 0.75, 1])

print("\nMin:", quartiles[0])
print("25th Percentile (Q1):", quartiles[0.25])
print("50th Percentile (Q2 or Median):", quartiles[0.5])
print("75th Percentile (Q3):", quartiles[0.75])
print("Max:", quartiles[1])

# Create a box plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x=df_complex['Values'], ax=ax)

# Set plot title and labels
ax.set_title("Box Plot with Quartiles")
ax.set_xlabel("Values")

# Show the plot
plt.show()
#%%
#Correlation; A measure of the strength and direction of a linear relationship between variables
import pandas as pd

# Sample data
data = {
    'Variable1': [1, 2, 3, 4, 5],
    'Variable2': [5, 4, 3, 2, 1],
    'Variable3': [2, 3, 1, 4, 5]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate correlation matrix
correlation_matrix = df.corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
#%%
#Exercise5:Create a DataFrame with two variables, A and B (The data follows a uniform distribution), and:

#Calculate the correlation coefficient between A and B.
#Visualize the correlation using a scatter plot.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with variables A and B
data = {'A': np.random.rand(100), 'B': np.random.rand(100)}
df = pd.DataFrame(data)

# Calculate Correlation Coefficient
correlation_coefficient = df['A'].corr(df['B'])
print(f'Correlation Coefficient: {correlation_coefficient}')

# Visualize Correlation Using Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='A', y='B', data=df)
plt.title('Scatter Plot (A, B)')
plt.xlabel('A')
plt.ylabel('B')
plt.show()

#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html
#https://ethanweed.github.io/pythonbook/03.01-descriptives.html






