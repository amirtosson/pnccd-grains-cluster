#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:49:02 2019

@author: tosson
"""
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import KMeans 
X= np.array([[1,2,1],
             [1,1,2],
             [5,8,3],
             [8,8,4],
             [1,6,5],
             [9,11,6],
             [8,3,4],
             [12,8,5],
             [6,1,6]])

#plt.scatter(X[:,0],X[:,1], s=150, linewidths=10)
#plt.show()

clf= KMeans(n_clusters=3)
clf.fit(X)

cetroids= clf.cluster_centers_
labels= clf.labels_

colors=["g.","r.","b.","c.","g.","r.","b.","c."]


fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax=ax.scatter(X[:,0],X[:,1],X[:,2], c='r', marker='o')






for i in range(len(X)):
    ax.plot(X[i][0],X[i][1], colors[labels[i]],markersize=25)

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y , Z = np.meshgrid(x, y , x)




ax.scatter(X,Y,Z, marker='x', s=150, linewidths=5)
plt.show()




def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(1, 10, 10)
y = np.linspace(21, 30, 10)
z = np.linspace(31, 40, 10)

X = np.meshgrid(x)

zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')



zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
fig = plt.figure()
fig=ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

plt.show()



ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z,c=Z, cmap='Greens')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');



x = np.array([1,1,1])
np.linalg.norm(x)






# generate 3 clusters of each around 100 points and one orphan point
N=100
data = numpy.random.randn(3*N,2)
data[:N] += 5
data[-N:] += 10
data[-1:] -= 20

# clustering
thresh = 1.5
clusters = hcluster.fclusterdata(data, thresh, criterion="distance")

print(clusters)
# plotting









