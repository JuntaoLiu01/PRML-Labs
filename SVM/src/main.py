# python: 2.7
# encoding: utf-8
from __future__ import print_function
import os
import numpy as np
from numpy import linalg

from plot import *
from svm  import *
from logistic import *
from linear import *
from kernel import *

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_data(fname):
    """
    载入数据。
    """
    fname = os.path.join(BASE_PATH,fname)
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """2/
    计算准确率。
    """
    return np.sum(label == pred) /(len(pred) * 1.0)

def task1(C=1.0,kernel=gaussian_kernel,sigma=0.1):
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_kernel.txt'
    test_file = 'data/test_kernel.txt'
    data_train = load_data(train_file)
    data_test = load_data(test_file)
    plot_data(data_train[:, :2],data_train[:, 2])

    svm = SVM(C=C,kernel=kernel,sigma=sigma)
    svm.train(data_train) 

    x_train = data_train[:, :2]   
    t_train = data_train[:, 2]  
    plot_contour(svm, x_train,t_train)
    t_train_pred = svm.predict(x_train) 
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

def task2():
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    #可视化数据
    print("visualizing data......")
    x_train = data_train[:,:2]
    t_train = data_train[:,2]
    plot_data(x_train,t_train)
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]

    #线性分类
    print("linear:")
    linear = LinearClassfication(verbose=True)
    linear.train(x_train, t_train)
    plot_boundery(linear.w,x_train,t_train,3)

    t_train_pred = linear.predict(x_train)
    t_test_pred = linear.predict(x_test)

    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

    # logistic 二分类
    print("logistic")
    logistic = LogisticRegression(verbose = True)
    logistic.train(x_train, t_train)
    plot_boundery(logistic.w,x_train,t_train,2)
    t_train_pred = logistic.predict(x_train,0.5)
    t_test_pred = logistic.predict(x_test,0.5)

    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))


    # SVM分类
    print("SVM:")
    C = 0.05
    kernel = linear_kernel
    svm = SVM(C=C,kernel=kernel)
    svm.train(data_train)

    plot_boundery(svm, x_train, t_train,1)
    t_train_pred = svm.predict(x_train)
    t_test_pred = svm.predict(x_test)

    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

def task3():

    train_file = 'data/train_multi.txt'
    test_file = 'data/test_multi.txt'
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    #可视化数据
    x_train = data_train[:,:2]
    t_train = data_train[:,2]
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    plot_data2(data_train,t_train)

    C = 0.05
    kernel = linear_kernel

    # svm1 
    svm1 = SVM(C=C,kernel = kernel)
    data1 = load_data(train_file)
    data1[data1[:,2] == 0,2] = -1
    data1[data1[:,2] == -1,2] = -1
    svm1.train(data1)

    x1_train = data1[:,:2]
    t1_train = data1[:,2]
    plot_contour(svm1,x1_train,t1_train)
    t1_train_pred = svm1.predict(x1_train)
    acc_train1 = eval_acc(t1_train, t1_train_pred)
    print("svm1 train accuracy: {:.1f}%".format(acc_train1 * 100))


    # svm2
    svm2 = SVM(C=C,kernel = kernel)
    data2 = load_data(train_file)
    data2[data2[:,2] == 1,2] = 0
    data2[data2[:,2] == -1,2] = 1
    data2[data2[:,2] == 0,2] = -1
    svm2.train(data2)

    x2_train = data2[:,:2]
    t2_train = data2[:,2]
    plot_contour(svm2,x2_train,t2_train)
    t2_train_pred = svm2.predict(x2_train)
    acc_train2 = eval_acc(t2_train, t2_train_pred)
    print("svm2 train accuracy: {:.1f}%".format(acc_train2 * 100))

    # svm3
    svm3 = SVM(C=C,kernel = kernel)
    data3 = load_data(train_file)
    data3[data3[:,2] == 1,2] = -1
    data3[data3[:,2] == 0,2] = 1
    svm3.train(data3)

    x3_train = data3[:,:2]
    t3_train = data3[:,2]
    plot_contour(svm3,x3_train,t3_train)
    t3_train_pred = svm3.predict(x3_train)
    acc_train3 = eval_acc(t3_train, t3_train_pred)
    print("svm3 train accuracy: {:.1f}%".format(acc_train3 * 100))

    t_train_pred = np.zeros(t_train.size)
    for i in range(x_train.shape[0]):
        t1 = np.dot(svm1.w,x_train[i]) + svm1.b
        t2 = np.dot(svm2.w,x_train[i]) + svm2.b
        t3 = np.dot(svm3.w,x_train[i]) + svm3.b

        if t1 >= t2:
            if t1 >= t3:
                t_train_pred[i] = 1
            else:
                t_train_pred[i] = 0
        else:
            if t2 >= t3:
                t_train_pred[i] = -1
            else:
                t_train_pred[i] = 0
    acc_train = eval_acc(t_train, t_train_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))

    t_test_pred = np.zeros(t_test.size)
    for i in range(x_test.shape[0]):
        t1 = np.dot(svm1.w,x_test[i]) + svm1.b
        t2 = np.dot(svm2.w,x_test[i]) + svm2.b
        t3 = np.dot(svm3.w,x_test[i]) + svm3.b

        if t1 >= t2:
            if t1 >= t3:
                t_test_pred[i] = 1
            else:
                t_test_pred[i] = 0
        else:
            if t2 >= t3:
                t_test_pred[i] = -1
            else:
                t_test_pred[i] = 0
    acc_test = eval_acc(t_test, t_test_pred)
    print("test accuracy: {:.1f}%".format(acc_test * 100))

if __name__ == '__main__':
    task1()
    # task2()
    # task3()
    print("Congratulations! You have done all of the works!\n")