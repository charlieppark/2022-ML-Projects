#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import numpy for matrix operations
#import sys for sys.maxsize
#import matplotlib for plot graph

import numpy as np
import sys
import matplotlib.pyplot as plt

#Generating data from data_hw1.csv

data = np.genfromtxt('data_hw1.csv', delimiter=',', skip_header=1)

X = data[:,0]
Y = data[:,1]

print(data)


# In[2]:


# To make contour plot for gradient descent, find global minimum of task1 by normal equation
# Normal Equation code is explained in task 2

X_0 = np.ones(X.shape[0])
X_1 = X

X_mat = np.c_[X_0, X_1]

XTX = X_mat.T@X_mat
XTX_1 = np.linalg.inv(XTX)
XTX_1XT = XTX_1@X_mat.T
theta = XTX_1XT@Y

Y_pred = theta[0] * X_0 + theta[1] * X_1

plt.figure(figsize=(30, 20))
plt.scatter(X,Y)
plt.scatter(X,Y_pred)
plt.show()

print(theta)
# [10.49320915 12.28806906] is global minimum


# In[3]:


#plot function is to plot function graph and contour together as subplot

def plot_function(x, y, func, theta, mse, history0, history1, batch_size, idx) :
  pred = func(theta, x)
  fig = plt.figure(figsize=(40, 20))
  
  ax1 = fig.add_subplot(1, 2, 1)
  
  ax2 = fig.add_subplot(1, 2, 2)

  ax1.scatter(x, y)
  ax1.scatter(x, pred)

  x1 = np.linspace(-1, 17, 700)
  x2 = np.linspace(-1, 17, 700)
  X1, X2 = np.meshgrid(x1, x2)
  Z = np.sqrt((X1-10.49320915)**2 + (X2-12.28806906)**2 ) # uses global minimum point to contour plot
  contours = ax2.contour(X1, X2, Z, 20)
  ax2.clabel(contours, inline = True, fontsize = 10)
  ax2.plot(history0, history1)
  ax2.plot(history0, history1, 'r*', label = "Cost function", linewidth=50, markersize=20)
  plt.show()
  #plt.savefig('figure'+f'{len(history0)}'.zfill(4))

    
#plot contour is to plot contour with different batches together (full batch, stochastic, mini-batch)
   
def plot_contour(x, y, func, theta, mse, history0, history1, batch_size, idx) :
  s = ''
  if batch_size == 0:
    s = 'g*'
  if batch_size == 1:
    s = 'b*'
  if batch_size == 30:
    s = 'r*'
  pred = func(theta, x)
  x1 = np.linspace(-2, 18, 1000)
  x2 = np.linspace(-2, 18, 1000)
  X1, X2 = np.meshgrid(x1, x2)
  Z = np.sqrt((X1-10.49320915)**2 + (X2-12.28806906)**2 )
  contours = plt.contour(X1, X2, Z, 20)
  plt.clabel(contours, inline = True, fontsize = 10)
  plt.plot(history0, history1)
  plt.plot(history0, history1, s, label = "Cost function", linewidth=10, markersize=5)
  plt.show()
  #plt.savefig(f'figure-batch{batch_size}' + f'--{len(history0)}'.zfill(4))
  #plt.savefig(f'{idx}'.zfill(4))


# In[4]:


# func is hypotheis h = θ0 + θ1 * X

def func(theta, X):
  return theta[0] + X * theta[1]


# gradient descent function gets parameter X, Y, func for necessary

# epsilon is to check convergence,
# If difference in error between k and k+1 iteration is smaller the epsilon, then stop iteration (convergence)

# batch_size_list is different batch sizes for gradient descent

# alpha is step size for gradient descent

# epoch is iteration of gradient descent

def gradient_descent(X, Y, func, epsilon=0.00001, batch_size_list=[0], alpha=0.05, epoch=351):
  plot_func = plot_function
  if len(batch_size_list) > 1: #if many batches are tested at a time
    plot_func = plot_contour
    epsilon = 0
  idx = 0
  for batch_size in batch_size_list:
    m = len(X) # full batch
    if batch_size > 0: # if batch_size, this function considers as full batch
      m = batch_size
    theta = [np.random.uniform(-1, 1) for i in range(2)] # initialize θ0and θ1
    error_before = -1 # check error of iteration before to know if convergence or not
    mse = sys.maxsize
    converge = False

    #histories of theta to draw on contour plot
    history0 = []
    history0.append(theta[0])
    history1 = []
    history1.append(theta[1])

    for i in range(epoch + 1) :
      if converge:
        break  

      # X_in and Y_in are X and Y of batches
      X_in = X
      Y_in = Y

      if m != len(X): # not full batch
        batch = np.random.randint(len(X), size=m) # randomly select batch samples
        X_in = X[batch]
        Y_in = Y[batch]
      
      Y_pred = func(theta, X_in)
      error = Y_in - Y_pred

      div = -(1/m)
    
      theta_diff = tuple(div * e for e in (np.sum(error), np.sum(X_in * error))) # calculate partial derivative
    
      theta[1] -= alpha * theta_diff[1]
      theta[0] -= alpha * theta_diff[0]

      error_before = mse
      mse = np.square(error).mean() #mean(squared(error))

      if m == len(X) and error_before - mse < epsilon and i > 50:
        converge = True
        print(f"θ0 = {theta[0]:4f}, θ1 = {theta[1]:4f}, epoch = {i}, error = {mse}")
        plot_func(X, Y, func, theta, mse, history0, history1, batch_size, idx)
    
      if i % 10 == 0 :
        print(f"θ0 = {theta[0]:4f}, θ1 = {theta[1]:4f}, epoch = {i}, error = {mse}")
        plot_func(X, Y, func, theta, mse, history0, history1, batch_size, idx)
    
      history0.append(theta[0])
      history1.append(theta[1])
      idx += 1

#gradient_descent(X, Y, func, batch_size_list=[0, 30, 1])
gradient_descent(X, Y, func)


# In[5]:


# Normal Equation

X_0 = np.ones(X.shape[0]) # X0 is ones
X_1 = X # X1 is X
X_2 = X**2 # X2 is X squared

X_mat = np.c_[X_0, np.c_[X_1, X_2]] # X_mat is [X0, X1, X2]

# variable names show intermediate steps to get normal equation
# @ is matrix multiplication
XTX = X_mat.T@X_mat
XTX_1 = np.linalg.inv(XTX) #inverse of XTX
XTX_1XT = XTX_1@X_mat.T
theta = XTX_1XT@Y

Y_pred = theta[0] * X_0 + theta[1] * X_1 + theta[2] * X_2

plt.figure(figsize=(30, 20))
plt.scatter(X,Y)
plt.scatter(X,Y_pred)
plt.savefig('normal equation')

print(theta)


# In[ ]:




