# encoding:utf-8
# python:2.7
from __future__ import division
from __future__ import print_function

import numpy as np
import math,random
import matplotlib.pyplot as plt

MARKER_COLOR = ['ro','go','bo','yo','ko']

def generate_data(mu,sigma,K=3,num=50):
    '''
    generate dataw which object to gaussian distribution
    '''
    X1 = []
    X2 = []
    for i in range(K):
        x1,x2 = np.random.multivariate_normal(mu[i],sigma[i],num).T
        X1.append(x1)
        X2.append(x2)
        plt.plot(x1,x2,MARKER_COLOR[i],markersize=5)
    plt.show()
    return X1,X2

def pad_data(X1,X2):
    '''
    reshape data to m by 1 size
    '''
    X1 = np.array(X1);X2 = np.array(X2)
    X1 = X1.reshape(X1.size)
    X2 = X2.reshape(X2.size)
    return X1,X2

def save_data(filename,X1,X2):
    '''
    save data on local disk
    '''
    with open(filename,'w') as wf:
        for i in range(X1.shape[0]):
            wf.write(str(X1[i]) + ' ')
            wf.write(str(X2[i]))
            wf.write('\n')
    
def load_data(filename):
    '''
    load data from local disk
    '''
    data = []
    with open(filename,'r') as rf:
        for line in rf:
            line = line.strip().split()
            x1 = float(line[0]);x2 = float(line[1])
            data.append([x1,x2])
        return np.array(data)

def plot_results(data,centroids,cluster_assign,K=3):
    '''
    plot the cluster result
    '''
    for i in range(K):
        points_inCluster = data[np.nonzero(cluster_assign[:, 0].A == i)[0]]
        plt.plot(points_inCluster[:,0],points_inCluster[:,1],MARKER_COLOR[i])
    cent_x1 = centroids[:,0]
    cent_x2 = centroids[:,1]
    plt.plot(cent_x1,cent_x2,'y+')
    plt.show()

def multi_gaussian(x,mu,sigma):
    '''
    the multi dimension form of gaussain distribution
    '''
    n = x.shape[0]
    mu = mu.reshape((1,n));x = x.reshape((1,n))
    expOn = float(-0.5 * (x - mu) * (sigma.I) * ((x - mu).T))
    divBy = pow(2 * np.pi, n / 2) * pow(np.linalg.det(sigma), 0.5)
    return pow(np.e, expOn) / divBy

def EM(data,max_iter=100,K=3):
    '''
    EM algorithm
    '''
    # initial params
    m,n = data.shape
    alphas = []
    for i in range(K):
        alphas.append(1.0/K)
    mu = [];index = []
    while len(mu) < K:
        i = int(math.floor(m*random.random()))
        while i in index:
            i = int(math.floor(m*random.random()))
        index.append(i)
        mu.append(data[i,:])
    sigma = [np.mat([[0.1, 0], [0, 0.1]]) for i in range(K)]
    gamma = np.mat(np.zeros((m, K)))

    # main process
    for i in range(max_iter):
        # compute gamma
        for j in range(m):
            sumAlphaMulP = 0
            for k in range(K):
                gamma[j,k] =  alphas[k] * multi_gaussian(data[j,:], mu[k], sigma[k])
                sumAlphaMulP += gamma[j,k]
            for k in range(K):
                gamma[j,k] = gamma[j,k]/sumAlphaMulP 
        sumGamma = np.sum(gamma, axis=0)

        # update alphas,mu,sigma
        for k in range(K):
            mu[k] = np.zeros((1, n))
            sigma[k] = np.mat(np.zeros((n, n)))
            for j in range(m):
                mu[k] += gamma[j,k] * data[j,:]
            mu[k] /= sumGamma[0, k]
            for j in range(m):
                sigma[k] += gamma[j, k] * (data[j, :] - mu[k]).T * (data[j, :] - mu[k])
            sigma[k] /= sumGamma[0, k]
            alphas[k] = sumGamma[0, k] / m
    return gamma

def GM(data,max_iter=100,K=3):
    '''
    the process of Mixture of Gaussian
    '''
    m,n = data.shape
    centroids = np.zeros((K, n))
    cluster_assign = np.mat(np.zeros((m, 2)))
    gamma = EM(data,max_iter,K)
    for i in range(m):
        cluster_assign[i,:] = np.argmax(gamma[i, :]), np.amax(gamma[i, :])
    for j in range(K):
        points_inCluster = data[np.nonzero(cluster_assign[:, 0].A == j)[0]]
        print(points_inCluster.shape)
        centroids[j, :] = np.mean(points_inCluster, axis=0) 
    return centroids, cluster_assign          

if __name__ == '__main__':
    # mu1 = [1.0,1.0]
    # sigma1 = [[0.5,0.0],[0.0,0.5]]
    # mu2 = [4.0,5.0]
    # sigma2 = [[0.4,0.0],[0.0,0.8]]
    # mu3 = [6.0,1.0]
    # sigma3 = [[0.6,0.0],[0.0,0.3]]
    # mu = [mu1,mu2,mu3];sigma = [sigma1,sigma2,sigma3]

    # X1,X2 = generate_data(mu,sigma,3)
    # X1,X2 = pad_data(X1,X2)
    # save_data('data.txt',X1,X2)
    data = load_data('data.txt')
    centroids,cluster_assign = GM(data,max_iter=100)
    plot_results(data,centroids,cluster_assign)


