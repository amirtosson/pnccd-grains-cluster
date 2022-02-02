#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:14:34 2019

@author: tosson
"""
# %%
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math as m 
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import KMeans 
import numpy as np
from pandas import DataFrame
from random import seed
from random import randint


seed(1)

#lattice constant
a = 5.653

####================================ helper functions ==============================
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return m.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return m.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def TransferToSpericalCoordinates(v):
    r = LA.norm(v)
    theta = m.atan(v[1]/v[0])
    phi = m.acos(v[2]/r)
    return theta, phi, r

def TransferToCartesianCoordinates(v,tre,pre):
    r = LA.norm(v)
    theta = m.degrees(m.atan(v[1]/v[0]))
    phi = m.degrees(m.acos(v[2]/r))
    qx = r * m.sin(m.radians(theta+tre))* m.cos(m.radians(phi+pre)) ;
    qy = r * m.sin(m.radians(theta+tre))* m.sin(m.radians(phi+pre));
    qz = r * m.cos(m.radians(theta+tre));
    return qx,qy,qz


def RotateSpericallyToRef(thetaRef,phiRef, rRef, theta,phi):
    theta= m.degrees(theta)
    phi= m.degrees(phi)
    qxRo = rRef * m.sin(m.radians(theta+thetaRef))* m.cos(m.radians(phi+phiRef));
    qyRo = rRef * m.sin(m.radians(theta+thetaRef))* m.sin(m.radians(phi+phiRef));
    qzRo = rRef * m.cos(m.radians(theta+thetaRef));
    return TransferToCartesianCoordinates([qxRo,qyRo,qzRo],theta,phi)
 

def Find_QxQyQz(h,k,l):

    qx = m.pi *  h / a ;
    qy = m.pi * k / a ;
    qz = m.pi * l / a ;
    return qx,qy,qz

def get_possible_hkl_permutations(h_range, k_range, l_range):
    all_values = []
    for h in range(-h_range,h_range):
        for k in range(-k_range,k_range):
            for l in range(-l_range,l_range):
                if h!=0 and k!=0 and l!=0: 
                    if ((h%2==0) and (k%2==0) and (l%2==0)) or ((h%2!=0) and (k%2!=0) and (l%2!=0)):
                        permutation = [h,k,l]
                        all_values.append(permutation)
    return np.array(all_values)

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])



# %%
# defining the h,k,l range
HKLRange = 20;
# number of simulated grains
number_grains = 50
hkl_array = get_possible_hkl_permutations(HKLRange, HKLRange, HKLRange)
load_orientation_data = True

if not load_orientation_data:
    # generating random rotation angles
    angle_orintations_phi = np.random.rand(number_grains) *100
    angle_orintations_theta = np.random.rand(number_grains) *100
else:
    # load rotation angles
    angle_orintations_phi = np.load('phi_'+str(number_grains)+'.npy')
    angle_orintations_theta = np.load('theta_'+str(number_grains)+'.npy')
    print("Data is loaded")

permutation_hkl_length = len(hkl_array)
# result arrays
ref_grain= np.zeros((permutation_hkl_length,9));
AllRes= np.zeros((number_grains,permutation_hkl_length,12));


for p in range(permutation_hkl_length):
    h, k, l = hkl_array[p,0], hkl_array[p,1], hkl_array[p,2]
    ref_grain[p,0] = h;
    ref_grain[p,1] = k;
    ref_grain[p,2] = l;
    ref_grain[p,3],ref_grain[p,4],ref_grain[p,5] = Find_QxQyQz(h,k,l);
    ref_grain[p,6],ref_grain[p,7],ref_grain[p,8] = TransferToSpericalCoordinates([ref_grain[p,3],ref_grain[p,4],ref_grain[p,5]])
    ref_grain[p,3],ref_grain[p,4],ref_grain[p,5]= RotateSpericallyToRef( ref_grain[p,6],ref_grain[p,7],ref_grain[p,8], 0,0)

            


all_res_random = [];
grain_ref_num = []
for g in range(number_grains):
    grain_data = []
    num_of_reflection = randint(3, 9)
    grain_ref_num.append(num_of_reflection)
    for t in range(num_of_reflection):
        theta= angle_orintations_phi[g]
        phi = angle_orintations_theta[g]   
        psi = np.random.rand(1) *100
        R = Rz(psi[0]) * Ry(theta) * Rx(phi)
        R = np.round(R, decimals=2)
        vec_rot = np.dot([ref_grain[t,0],ref_grain[t,1],ref_grain[t,2]],R)   
        any_reflection = randint(0, permutation_hkl_length)
        q_x, q_y, q_z = RotateSpericallyToRef(ref_grain[t,3],ref_grain[t,4],ref_grain[t,5],theta,phi)
        
        v1=[ref_grain[t,3],ref_grain[t,4]];
        v2=[q_x,q_y]; 
        alpha = m.degrees(angle(v1, v2))#+randint(-20, 20)/100
        #y-z 
        v1=[ref_grain[t,4],ref_grain[t,5]];
        v2=[q_y,q_z]; 
        beta= m.degrees(angle(v1, v2))#+randint(-20, 20)/100
        #z-x 
        v1=[ref_grain[t,5],ref_grain[t,3]];
        v2=[q_z,q_x]; 
        gamma= m.degrees(angle(v1, v2))#+randint(-20, 20)/100
        
        
        #print(str(num_of_reflection)+ " "+ str(any_reflection))
        data_ref = [g+1,ref_grain[t,3],ref_grain[t,4],ref_grain[t,5],vec_rot[0],vec_rot[1],vec_rot[2], alpha, beta, gamma ]
        grain_data.append(data_ref)
    grain_data = np.array(grain_data)
    all_res_random.append(grain_data)
        

all_res_random = np.array(all_res_random)


colors_array = []

for _ in range(number_grains):
    col = '#%06X' % randint(0, 0xFFFFFF)
    colors_array.append(col)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot(ref_grain[:,0],ref_grain[:,1],ref_grain[:,2], c='red',markerfacecolor='None', linestyle = 'None',marker='o',label='Ref')
for g in range(number_grains):
    col = '#%06X' % randint(0, 0xFFFFFF)
    ax.plot(all_res_random[g][:,7],all_res_random[g][:,8],all_res_random[g][:,9], c=col,markerfacecolor='None', linestyle = 'None',marker='o',label='Grain_'+str(g+1))


ax.set_xlabel('alpha[°]', fontsize=5, labelpad=-10)
ax.set_ylabel('beta[°]', fontsize=5, labelpad=-10)
ax.set_zlabel('gamma[°]', fontsize=5, labelpad=-10)
ax.tick_params(axis='both', which='major', labelsize=6, pad=-4)
ax.tick_params(axis='both', which='minor', labelsize=6, pad=-4)
#plt.legend(loc="upper left", fontsize=5)
#plt.savefig('grains_'+str(number_grains)+'.eps', format='eps', dpi=1200)
plt.savefig('grains_random_'+str(number_grains)+'.png', format='png', dpi=1200)

# %%

fig = plt.figure(figsize = (10, 5))
x_data = range(1,number_grains+1)
# creating the bar plot
plt.bar(x_data , grain_ref_num, color ='blue', width =.5)
plt.xticks(np.arange(min(x_data )+1, max(x_data )+1, 1.0 ),fontsize=10, rotation=90)
plt.xlabel("Grain ID ")
plt.ylabel("No. of reflections")
plt.savefig('reflections_distribution_'+str(number_grains)+'.png', format='png', dpi=800)



fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot(ref_grain[:,0],ref_grain[:,1],ref_grain[:,2], c='red',markerfacecolor='None', linestyle = 'None',marker='o',label='Ref')
for g in range(number_grains):

    col = '#%06X' % randint(0, 0xFFFFFF)
    ax.plot(all_res_random[g][:,4],all_res_random[g][:,5],all_res_random[g][:,6], c=col,markerfacecolor='None', linestyle = 'None',marker='o',label='Grain_'+str(g+1))

ax.set_xlabel('qx[A°]', fontsize=5, labelpad=-10)
ax.set_ylabel('qy[A°]', fontsize=5, labelpad=-10)
ax.set_zlabel('qz[A°]', fontsize=5, labelpad=-10)
ax.tick_params(axis='both', which='major', labelsize=6, pad=-4)
ax.tick_params(axis='both', which='minor', labelsize=6, pad=-4)
#plt.legend(loc="upper left", fontsize=5)
#plt.savefig('receprocal_space_'+str(number_grains)+'.eps', format='eps')
plt.savefig('receprocal_space_random_'+str(number_grains)+'.png', format='png', dpi=1200)




#===================== Clustering ======================
allDataPoints = np.concatenate((all_res_random), axis=0)

Data = {'x': allDataPoints[:,7],
        'y': allDataPoints[:,8],
        'z': allDataPoints[:,9]
      }

df = DataFrame(Data,columns=['x','y','z']) 
df = df.sample(frac=1)
fig3 = plt.figure()
thresh = 0.9
clusters = hcluster.fclusterdata(df, thresh, criterion="distance")
number_of_cluster=np.amax(clusters); 


print("Num of cluster (intial)= " + str(number_of_cluster))

Z = hcluster.linkage(df, method='ward')

dend = hcluster.dendrogram(hcluster.linkage(df/10, method='ward'), leaf_rotation=90.,leaf_font_size=12.,  show_contracted=True,labels= None, p=12, truncate_mode='lastp', show_leaf_counts=True,)
plt.axhline(y = thresh, color = 'r', linestyle = '--', label = "Threshold")
plt.xlabel('Sample index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.legend(loc="upper right", fontsize=12)
#plt.savefig('hc_dend'+str(number_grains)+'.eps', format='eps')
plt.savefig('hc_dendrandom_'+str(number_grains)+'.png', format='png', dpi=1200)

n=0;
clus = number_of_cluster - 1;
intertia = 3
ino = []
chosen_index = 0
fulfilled = False
normalized_gradient = []
while True: 
    clf= KMeans(n_clusters=clus)
    clf.fit(df)
    labels= clf.labels_
    intertia = clf.inertia_/(len(labels) * 5)
    ino.append(intertia)
    normalized_gradient.append(intertia)

    if fulfilled: 
        break
    if n>0: 
        n_g = np.sum(np.gradient(ino))/len(ino)

        if n_g < 1:
            fulfilled = True
            normalized_gradient.append(n_g)
            chosen_index = n

    n = n+1;
    clus = clus + 1;

fig4 = plt.figure()   
x_valuse = range(number_of_cluster - 1 , clus + 1)

see_values = np.array(normalized_gradient)

plt.plot(number_of_cluster-1 +chosen_index , see_values[chosen_index] , marker='X',markersize=20,label='Elbow point',  mew=0.1)  
plt.xticks(np.arange(min(x_valuse), max(x_valuse)+1, 1.0))   
plt.plot(x_valuse, see_values , marker='o',markersize=8)  
plt.legend(loc="upper right")
plt.xlabel('Number of clusters', fontsize=12)
plt.ylabel('Normalized SEE', fontsize=12)
#plt.savefig('elbow_'+str(number_grains)+'.eps', format='eps')
plt.savefig('elbow_random_'+str(number_grains)+'.png', format='png', dpi=1200)


fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.plot(df['x'],df['y'],df['z'], linestyle = 'None', marker='o',markersize=12, label = "data points" )
ax2.plot(clf.cluster_centers_[:, 0],clf.cluster_centers_[:, 1],clf.cluster_centers_[:, 2], linestyle = 'None', c='red', marker='X',  label = "centroids")
ax2.set_xlabel('alpha[°]', fontsize=5, labelpad=-10)
ax2.set_ylabel('beta[°]', fontsize=5, labelpad=-10)
ax2.set_zlabel('gamma[°]', fontsize=5, labelpad=-10)
ax2.tick_params(axis='both', which='major', labelsize=6, pad=-4)
ax2.tick_params(axis='both', which='minor', labelsize=6, pad=-4)
#plt.savefig('clusters_3d_'+str(number_grains)+'.eps', format='eps')
plt.legend(loc="upper left", fontsize=8)
plt.savefig('clusters_3d_random_'+str(number_grains)+'.png', format='png', dpi=1200)
