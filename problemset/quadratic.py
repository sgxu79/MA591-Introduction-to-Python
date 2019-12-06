#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 02:25:54 2019

@author: 15306
"""

import numpy as np
import matplotlib.pyplot as plt

class Quadratic:
    """
    Quadratic function taking 3 parameters. 
    a,b,c are second order, first order and constant paramter.
    """

    def __init__(self, a = 1, b = 0, c = 0):
        self.a = a
        self.b = b
        self.c = c
    
    def value(self, x):
        """
        Compute quadratic function at a point x
        """
        return self.a * x ** 2 + self.b * x + self.c
        
    def plot(self, L, R, n):
        """
        Plot the quadratic function in the interval [L, R] at 
        n evenly spaced points.
        """
        x = np.linspace(L, R, n)
        y = self.value(x)
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Quadratic Function with a={}, b={}, c={}".format(self.a, self.b, self.c))
    
    def roots(self):
        """
        Return roots of the quadratic function
        """
        delta = self.b ** 2 - 4 * self.a * self.c
        return ((-self.b - delta ** 0.5) / (2 * self.a), (-self.b + delta ** 0.5) / (2 * self.a))