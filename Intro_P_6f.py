#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 08:57:48 2023

@author: maryamramezaniziarani
"""

#%%
#Inheritance
#%%
# superclass
class Vehicle:
    def __init__(self, brand, model): # Constructor method initializes attributes
        self.brand = brand
        self.model = model

    def start_engine(self):
        return "Engine started"

    def stop_engine(self):
        return "Engine stopped"


# Subclass1
class Car(Vehicle):
    def __init__(self, brand, model, num_doors):
        # Call the constructor of the superclass using super()
        super().__init__(brand, model)
        self.num_doors = num_doors

    def start_engine(self):
        # Override the start_engine method for Car
        return "Car engine started"


# Subclass2
class Bicycle(Vehicle):
    def __init__(self, brand, model, num_gears):
        # Call the constructor of the superclass using super()
        super().__init__(brand, model)
        self.num_gears = num_gears

    def start_engine(self):
        # Override the start_engine method for Bicycle
        return "Bicycle doesn't have an engine"


# Create instances of the classes
car = Car(brand="Porsche", model="Panamera", num_doors=4)
bicycle = Bicycle(brand="Schwinn", model="Roadster", num_gears=7)

# Access attributes and call methods
print(f"{car.brand} {car.model} with {car.num_doors} doors: {car.start_engine()}")
print(f"{bicycle.brand} {bicycle.model} with {bicycle.num_gears} gears: {bicycle.start_engine()}")
#%%
#Exercise1:
# Imagine you are tasked with designing a system to represent geometric shapes. Each shape has common
# properties (e.g., area, perimeter), but different shapes may have unique properties and behaviors. 
# Your goal is to create a class hierarchy to represent various geometric shapes.

# Requirements:
# Define a base class called Shape with the following methods:
# - area(): Returns the area of the shape.
# - perimeter(): Returns the perimeter of the shape.

# Implement three subclasses: Circle, Rectangle, and Triangle. Each subclass should inherit from the Shape class.

# For each subclass, implement the necessary methods to calculate the area and perimeter based on their specific formulas:
# - Circle: area = πr^2, perimeter = 2πr
# - Rectangle: area = length × width, perimeter = 2(length + width)
# - Triangle: Use Heron's formula to calculate the area, perimeter = side1 + side2 + side3

# Create instances of each shape and demonstrate the use of their methods.

# Add error handling to ensure that the sides, radius, length, and width are non-negative values.

import math

# Base class
class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement this method")

    def perimeter(self):
        raise NotImplementedError("Subclasses must implement this method")

# Subclasses
class Circle(Shape):
    def __init__(self, radius):
        if radius < 0:
            raise ValueError("Radius must be a non-negative value.")
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, length, width):
        if length < 0 or width < 0:
            raise ValueError("Length and width must be non-negative values.")
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * (self.length + self.width)

class Triangle(Shape):
    def __init__(self, side1, side2, side3):
        if side1 < 0 or side2 < 0 or side3 < 0:
            raise ValueError("All sides of the triangle must be non-negative values.")
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3

    def area(self):
        s = (self.side1 + self.side2 + self.side3) / 2
        return math.sqrt(s * (s - self.side1) * (s - self.side2) * (s - self.side3))

    def perimeter(self):
        return self.side1 + self.side2 + self.side3

# Example Usage
try:
    circle = Circle(radius=-5)
    rectangle = Rectangle(length=4, width=6)
    triangle = Triangle(side1=3, side2=4, side3=5)

    # Demonstrate methods
    print(f"Circle Area: {circle.area()}, Perimeter: {circle.perimeter()}")
    print(f"Rectangle Area: {rectangle.area()}, Perimeter: {rectangle.perimeter()}")
    print(f"Triangle Area: {triangle.area()}, Perimeter: {triangle.perimeter()}")

except ValueError as e:
    print(f"Error: {e}")

#%%
#Python Magic Methods
#%%
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def start_engine(self):
        return "Engine started"

    def stop_engine(self):
        return "Engine stopped"

    def __str__(self):
        return f"{self.brand} {self.model}"

    def __repr__(self):
        return f"Vehicle(brand={self.brand}, model={self.model})"


class Car(Vehicle):
    def __init__(self, brand, model, num_doors):
        super().__init__(brand, model)
        self.num_doors = num_doors

    def start_engine(self):
        return "Car engine started"

    def __str__(self):
        return f"{super().__str__()} with {self.num_doors} doors"

    def __repr__(self):
        return f"Car(brand={self.brand}, model={self.model}, num_doors={self.num_doors})"


class Bicycle(Vehicle):
    def __init__(self, brand, model, num_gears):
        super().__init__(brand, model)
        self.num_gears = num_gears

    def start_engine(self):
        return "Bicycle doesn't have an engine"

    def __str__(self):
        return f"{super().__str__()} with {self.num_gears} gears"

    def __repr__(self):
        return f"Bicycle(brand={self.brand}, model={self.model}, num_gears={self.num_gears})"


# Create instances of the classes
car = Car(brand="Porsche", model="Panamera", num_doors=4)
bicycle = Bicycle(brand="Schwinn", model="Roadster", num_gears=7)

# Access attributes and call methods
print(str(car))
print(str(bicycle))

# For debugging purposes
print(repr(car))
print(repr(bicycle))
#%% Exercise2:

# 1. NoLengthClass
class NoLengthClass:
    pass

# a. create an object obj_no_length of this class.
obj_no_length = NoLengthClass()

# b. Try to use the len() function on the obj_no_length object. Observe the error (try-except)
try:
    len_obj_no_length = len(obj_no_length)
except TypeError as e:
    len_obj_no_length = f"TypeError: {e}"

# 2. LengthDefined Class
class LengthDefined:
    def __len__(self):
        return 2

# a. create an object obj_length_defined of this class.
obj_length_defined = LengthDefined()

# b. Use the len() function on the obj_length_defined object and observe the result.
len_obj_length_defined = len(obj_length_defined)

# 3. General Questions
# a. Explain the purpose of the __len__ dunder method in Python.
purpose_of_len_dunder = "The __len__  method is used to define the custom behavior of the len() function for instances of a class."

# b. Why does the NoLengthClass raise a TypeError when len(obj_no_length) is called?
reason_for_type_error = "The NoLengthClass raises a TypeError because it does not define the __len__ method. The len() function requires this method to operate on objects."

# c. How does adding the __len__ dunder method to the LengthDefined class resolve the TypeError?
resolution_of_type_error = "Adding the __len__  method to the LengthDefined class provides a valid implementation for len(), resolving the TypeError."

# 4. Exercise Extension
# a. DynamicLength Class
class DynamicLength:
    def __init__(self, elements):
        self.elements = elements

    def __len__(self):
        return len(self.elements)

# b. create an object dynamic_obj of the DynamicLength class with a list of elements.
dynamic_obj = DynamicLength(elements=[1, 2, 3, 4, 5,7])

# c. Use the len() function on the dynamic_obj object and observe the result.
len_dynamic_obj = len(dynamic_obj)

# Results
print("1. NoLengthClass:")
print(f"a. {obj_no_length}")
print(f"b. {len_obj_no_length}\n")

print("2. LengthDefined Class:")
print(f"a. {obj_length_defined}")
print(f"b. {len_obj_length_defined}\n")

print("3. General Questions:")
print(f"a. {purpose_of_len_dunder}")
print(f"b. {reason_for_type_error}")
print(f"c. {resolution_of_type_error}\n")

print("4. Exercise Extension:")
print(f"a. {dynamic_obj}")
print(f"c. {len_dynamic_obj}")
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)

