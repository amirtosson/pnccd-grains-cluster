#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:43:27 2019

@author: tosson
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import math 
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import KMeans 
import numpy as np
a = 5.65312212


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def Find_QxQyQz(h,k,l):

    qx = math.pi *  h / a ;
    qy = math.pi * k / a ;
    qz = math.pi * l / a ;
    return qx,qy,qz

def RotateReciprocalCoordinates(qx,qy,qz,alpha,beta,gamma):
    
    #x-y plane around z (gamma)  
    qxRo = qx * math.cos(math.radians(gamma)) - qy * math.sin(math.radians(gamma));
    qyRo = qx * math.sin(math.radians(gamma)) + qy * math.cos(math.radians(gamma));
    qzRo = qz;
    #y-z plane around x (alpha)    
    qxRo = qxRo;
    qyRo = qyRo * math.cos(math.radians(alpha)) -  qzRo * math.sin(math.radians(alpha));
    qzRo = qyRo * math.sin(math.radians(alpha)) +  qzRo * math.cos(math.radians(alpha));
    #z-x plane around y (beta)   
    qxRo = qxRo * math.cos(math.radians(beta)) +  qzRo * math.sin(math.radians(beta));
    qyRo = qyRo;
    qzRo = -qxRo * math.sin(math.radians(beta)) +  qzRo * math.cos(math.radians(beta));
    return qxRo,qyRo,qzRo

HKLRange = 3;
count=0;
for h in range(-HKLRange,HKLRange):
    for k in range(-HKLRange,HKLRange):
        for l in range(-HKLRange,HKLRange):
            if h!=0 and k!=0 and l!=0: 
                if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                    count = count+1;

AllRes= np.zeros((count,6));
count=0;
for h in range(-HKLRange,HKLRange):
    for k in range(-HKLRange,HKLRange):
        for l in range(-HKLRange,HKLRange):
            if h!=0 and k!=0 and l!=0: 
                if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                    AllRes[count,0] = h;
                    AllRes[count,1] = k;
                    AllRes[count,2] = l;
                    AllRes[count,3],AllRes[count,4],AllRes[count,5] = Find_QxQyQz(h,k,l);
                    count = count+1;
                    
                    
################################ new grain with orientation ########################################
alpha =0;
beta = 0;
gamma =40; 


count = 0;
for h in range(-HKLRange,HKLRange):
    for k in range(-HKLRange,HKLRange):
        for l in range(-HKLRange,HKLRange):
            if h!=0 and k!=0 and l!=0: 
                if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                    count = count+1;

AllRes2= np.zeros((count,6));
count=0;
for h in range(-HKLRange,HKLRange):
    for k in range(-HKLRange,HKLRange):
        for l in range(-HKLRange,HKLRange):
            if h!=0 and k!=0 and l!=0: 
                if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                    AllRes2[count,0] = h;
                    AllRes2[count,1] = k;
                    AllRes2[count,2] = l;
                    AllRes2[count,3],AllRes2[count,4],AllRes2[count,5] = Find_QxQyQz(h,k,l);
                    AllRes2[count,3],AllRes2[count,4],AllRes2[count,5] = RotateReciprocalCoordinates(AllRes2[count,3],AllRes2[count,4],AllRes2[count,5],alpha,beta,gamma);
                    count = count+1;


                    
                        
for p in range(10):
    v1= [AllRes2[p,4],AllRes2[p,3]];
    v2= [AllRes[p,4],AllRes[p,3]];
    an = angle(v1,v2);
    print('y',an)



################################ new grain with orientation ########################################

alpha =0;
beta = 0;
gamma =20; 


count = 0;
for h in range(-HKLRange,HKLRange):
    for k in range(-HKLRange,HKLRange):
        for l in range(-HKLRange,HKLRange):
            if h!=0 and k!=0 and l!=0: 
                if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                    count = count+1;

AllRes3= np.zeros((count,6));
count=0;
for h in range(-HKLRange,HKLRange):
    for k in range(-HKLRange,HKLRange):
        for l in range(-HKLRange,HKLRange):
            if h!=0 and k!=0 and l!=0: 
                if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                    AllRes3[count,0] = h;
                    AllRes3[count,1] = k;
                    AllRes3[count,2] = l;
                    AllRes3[count,3],AllRes3[count,4],AllRes3[count,5] = Find_QxQyQz(h,k,l);
                    AllRes3[count,3],AllRes3[count,4],AllRes3[count,5] = RotateReciprocalCoordinates(AllRes3[count,3],AllRes3[count,4],AllRes3[count,5],alpha,beta,gamma);
                    count = count+1;


                    
                        
for p in range(10):
    v1= [AllRes3[p,4],AllRes3[p,3]];
    v2= [AllRes[p,4],AllRes[p,3]];
    an = angle(v1,v2);
    print('y',an)




################################ new grain with orientation ########################################


alpha =0;
beta = 0;
gamma =56; 


count = 0;
for h in range(-HKLRange,HKLRange):
    for k in range(-HKLRange,HKLRange):
        for l in range(-HKLRange,HKLRange):
            if h!=0 and k!=0 and l!=0: 
                if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                    count = count+1;

AllRes4= np.zeros((count,6));
count=0;
for h in range(-HKLRange,HKLRange):
    for k in range(-HKLRange,HKLRange):
        for l in range(-HKLRange,HKLRange):
            if h!=0 and k!=0 and l!=0: 
                if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                    AllRes4[count,0] = h;
                    AllRes4[count,1] = k;
                    AllRes4[count,2] = l;
                    AllRes4[count,3],AllRes4[count,4],AllRes4[count,5] = Find_QxQyQz(h,k,l);
                    AllRes4[count,3],AllRes4[count,4],AllRes4[count,5] = RotateReciprocalCoordinates(AllRes4[count,3],AllRes4[count,4],AllRes4[count,5],alpha,beta,gamma);
                    count = count+1;


                    
                        
for p in range(10):
    v1= [AllRes4[p,4],AllRes4[p,3]];
    v2= [AllRes[p,4],AllRes[p,3]];
    an = angle(v1,v2);
    print('y',an)    
                    
                    