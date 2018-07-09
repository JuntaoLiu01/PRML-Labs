# python: 2.7
# encoding: utf-8
import numpy as np
import math
import random
from numpy import linalg
import cvxopt
import cvxopt.solvers
from kernel import *

class SVM(object):
    def __init__(self,C=None,kernel=linear_kernel,sigma=0.1,tol=0.001,max_passes=5):
         self.tol = tol
         self.max_passes = max_passes
         self.kernel = kernel
         self.C = C 
         self.sigma = sigma
         if self.C is not None:
             self.C = float(self.C)

    def train(self,data_train):
      
        X = data_train[:,:2]
        t = data_train[:,2] 
        m = X.shape[0]
        n = X.shape[1]

        # Variables
        alphas = np.zeros(m)
        b = 0
        E = np.zeros(m)
        passes = 0
        eta = 0
        L = 0
        H = 0

        # compute kernel
        if self.kernel is linear_kernel:
            K = np.dot(X,X.T)
        elif self.kernel is gaussian_kernel:
            temp = np.sum(np.square(X),axis=1)
            X2 = temp
            temp = -2 * np.dot(X,X.T)
            temp = temp[...,:] + X2
            K = temp[:,...] + X2.reshape((m,1))
            K = np.power(self.kernel(1,0,self.sigma),K)
        else:
            K = np.zeros((m,m))
            for i in range(m):
                for j in range(i,m):
                    K[i,j] = self.kernel(np.array(X[i]), np.array(X[j]).reshape((m,1)))
                    K[j,i] = K[i,j]

        # training
        print("start training...")
        dots = 12
        while passes < self.max_passes:
            num_changed_alphas  = 0
            for i in range(m):
                # Calculate Ei = f(x(i)) - y(i)
                # E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
                E[i] = b + np.sum(np.multiply(np.multiply(alphas,t),K[:,i])) - t[i]
                if (t[i] * E[i] < -self.tol and alphas[i] < self.C) or (t[i] * E[i] > self.tol and alphas[i] > 0):
                    # In practice, there are many heuristics one can use to select
                    # the i and j. In this simplified code, we select them randomly
                    j = math.ceil(random.random()) 
                    while j == i:
                        j = math.floor(m*random.random())
                    j = int(j)
                    # Calculate Ej = f(x(j)) - y(j)
                    E[j] = b + np.sum(np.multiply(np.multiply(alphas,t),K[:,j])) - t[j]
                    
                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    # compute L and H
                    if t[i] == t[j]:
                        L = max(0,alphas[j] + alphas[i]- self.C)
                        H = min(self.C,alphas[j] + alphas[i])
                    else:
                        L = max(0,alphas[j] - alphas[i])
                        H = min(self.C,self.C + alphas[j] - alphas[i])

                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0:
                        continue
                    
                    # Compute and clip new value for alpha j
                    alphas[j] = alphas[j] - (t[j] * (E[i] - E[j])) /float(eta)

                    alphas[j] = min(H, alphas[j])
                    alphas[j] = max(L, alphas[j])

                    # Check if change in alpha is significant
                    if abs(alphas[j] - alpha_j_old) < self.tol:
                        alphas[j] = alpha_j_old
                        continue
                    # Determine value for alpha i
                    alphas[i] = alphas[i] + t[i]*t[j]*(alpha_j_old - alphas[j])
                    # Compute b1 and b2
                    b1 = b-E[i]-t[i]*(alphas[i]-alpha_i_old)*K[i][j]-t[j]*(alphas[j]-alpha_j_old)*K[i][j]
                    b2 = b-E[j]-t[i]*(alphas[i]-alpha_i_old)*K[i][j]-t[j]*(alphas[j]-alpha_j_old)*K[j][j]

                    # compute b
                    if 0 < alphas[i] and alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] and alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1+b2)/2
                    
                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes = passes+1
            else: 
                passes = 0

            dots = dots + 1
            if dots > 78:
                dots = 0

        print("Done")
        idx = []
        for i in range(m):
            if alphas[i] > 0:
                idx.append(i)
        # idx = alphas > 0
        self.X = X[idx,:]
        self.t = t[idx]
        self.alphas = alphas[idx]
        self.b = b
        self.w = np.dot(np.multiply(alphas,t),X).T

    def predict(self,X):
        m = X.shape[0]
        p = np.zeros(m)
        pred = np.zeros(m)
        if self.kernel is linear_kernel:
            p = np.dot(X,self.w) + self.b
        elif self.kernel is gaussian_kernel:
            X1 = np.sum(np.square(X),axis=1)
            X1 = X1.reshape((m,1))
            X2 = np.sum(np.square(self.X),axis=1)
            K = -2 * np.dot(X,self.X.T)
            K = K[...,:] + X2;K = K[:,...] + X1
            K = np.power(self.kernel(1,0,self.sigma),K)
            K = K[...,:] * self.t;K = K[...,:] * self.alphas
            p = np.sum(K,axis=1)
        else:
            for i in range(m):
                prediction = 0
                for j in range(self.X.shape[0]):
                    prediction = prediction + self.alphas[j] * self.t[j] * self.kernel(np.array([self.X[i]]),np.array([self.X[j].reshape(m,1)]))
                p[i] = prediction + self.b  
        l_pos = []
        l_neg = []
        for i in range(m):
            if p[i] >= 0:
                l_pos.append(i)
            else:
                l_neg.append(i)

        pred[l_pos] = 1
        pred[l_neg] = -1
        return pred

    def cost(self,y,t):
        m = y.size
        t1 = 1-np.multiply(y,t)
        t1[t1<0] = 0
        
        t2 = 0
        if self.C is not None:
            t2 = 1/2.0/self.C * np.square(self.w)
        return t1.mean() + 1/2.0 * np.sum(t2)/m
