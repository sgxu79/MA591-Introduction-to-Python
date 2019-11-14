# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:24:40 2019

@author: 15306
"""
from math import pi, sqrt

class Ellipse:
    """
    Ellipse centered at x0, y0 with wdith and heigh 2a, 2b is defined by 
    the equation(x-x0)^2/a^2 + (y-y0)^2/b^2 = 1
    """
    
    def __init__(self, x0, y0, a, b):
        self.x0, self.y0, self.a, self.b = x0, y0, a, b
        
    def area(self):
        """
        A = pi*a*b 
        """
        return pi*self.a*self.b
    
    def circumference(self):
        """
        Approximation by Ramanujan, exact for circles
        """
        h = (self.a-self.b)**2/(self.a+self.b)**2
        return pi*(self.a+self.b)*(1+3*h/(10+sqrt(4-3*h)))
    
class Circle(Ellipse):
    """
    Circle with radius r centered at x0, y0 is defined by the equation
    (x-x0)^2+(y-y0)^2=r^2
    """
    
    def __init__(self, x0, y0, r):
        self.r = r
        Ellipse.__init__(self, x0, y0, a=self.r, b=self.r)
        



el = Ellipse(0,0,1,2)


print('A ellipse with width %.12g and height %.12g at (%.12g, %.12g) has area %.12g and perimeter %.12g'% \
      (2*el.a, 2*el.b, el.x0, el.y0, el.area(), el.circumference())) 


c = Circle(1,2,2)


print('A circle with radius %.12g at (%.12g, %.12g) has area %.12g and perimeter %.12g'% \
      (c.r, c.x0, c.y0, c.area(), c.circumference())) 
