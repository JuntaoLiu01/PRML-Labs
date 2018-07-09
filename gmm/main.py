# encoding:utf-8
# python:2.7
from __future__ import print_function

from utils import *

def run(max_iter=100,generate_new_data=False,**kw):
    '''
    start the program
    '''
    if generate_new_data:
        try:
            mu = kw['mu']
            sigma = kw['sigma']
            K = kw['K']
        except Exception as e:
            print(e)
        else:
            if 'num' in kw.keys():
                num = kw['num']
                X1,X2 = generate_data(mu,sigma,K,num)
            else:
                X1,X2 = generate_data(mu,sigma,K)
            X1,X2 = pad_data(X1,X2)
            save_data('data.txt',X1,X2)
            data = load_data('data.txt')
            centroids,cluster_assign = GM(data,max_iter=max_iter,K=K)
            plot_results(data,centroids,cluster_assign,K=K)
    else:
        data = load_data('data.txt')
        centroids,cluster_assign = GM(data,max_iter=max_iter)
        plot_results(data,centroids,cluster_assign)

if __name__ == '__main__':
    run(max_iter=3)
