import os
import sys
sys.path.append(os.path.abspath("./libsvm/python"))
# print(sys.path)
from svm import *
from svmutil import *

import numpy as np
import matplotlib.pyplot as plt

def circle(radius, sigma=0, num_points=50):
    t = np.linspace(0, 2*np.pi, num_points)
    d = np.zeros((num_points,2), dtype=np.float)
    d[:,0] = radius*np.cos(t) + np.random.randn(t.size)*sigma
    d[:,1] = radius*np.sin(t) + np.random.randn(t.size)*sigma
    return d


num_train = 100
num_test = 30
sigma = 0.2

d1 = circle(3, sigma, num_train)
d2 = circle(5, sigma, num_train)

plt.figure()
plt.plot(d1[:,0],d1[:,1],'ro')
plt.plot(d2[:,0],d2[:,1],'bo')
plt.show()