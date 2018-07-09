import numpy as np
import matplotlib.pyplot as plt

def plot_data(X,t):
    pos = np.where(t == 1)
    neg = np.where(t == -1)

    plt.plot(X[pos,0],X[pos,1], 'r+',markersize=5)
    plt.plot(X[neg,0],X[neg,1], 'bo',markersize=5)
    # plt.plot(x_test, y_test, 'k')
    # plt.plot(x_test, y_test_pred)
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.legend(['x1', 'x2'])
    plt.show()

def plot_data2(X,t):
    pos = np.where(t == 1)
    neg = np.where(t == -1)
    zero = np.where(t == 0)

    plt.plot(X[pos,0],X[pos,1], 'r+',markersize=5)
    plt.plot(X[neg,0],X[neg,1], 'bo',markersize=5)
    plt.plot(X[zero,0],X[zero,1],'k^',markersize=5)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def plot_boundery(model,X,y,type=1):
    X_pos = X[y==1]
    X_neg = X[y==-1]

    def f(x,w,b,c=0):
        return (-w[0]*x -b +c)/w[1]

    x1_min = np.min(X[:,0])+20
    x1_max = np.max(X[:,0])-20
    plt.plot(X_pos[:,0],X_pos[:,1],"r+")
    plt.plot(X_neg[:,0],X_neg[:,1],"bo")
    # plt.scatter(svm.sv[:,0],svm.sv[:,1],s=100,c="g")

    if type == 1:
        # w.x + b = 0
        a0 = x1_min; a1 = f(a0, model.w,model.b)
        b0 = x1_max;  b1 = f(b0,model.w,model.b)
        plt.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = x1_min; a1 = f(a0, model.w,model.b,1)
        b0 = x1_max;  b1 = f(b0,model.w,model.b,1)
        plt.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = x1_min; a1 = f(a0, model.w,model.b,-1)
        b0 = x1_max;  b1 = f(b0,model.w,model.b,-1)
        plt.plot([a0,b0], [a1,b1], "k--")

        plt.axis("tight")
    elif type == 2:
        arg_w = model[1:]
        arg_b = model[0]
        a0 = x1_min; a1 = f(a0,arg_w,arg_b)
        b0 = x1_max;  b1 = f(b0,arg_w,arg_b)
        plt.plot([a0,b0], [a1,b1], "k")

    else:
        def f2(x,w1,w2,b1,b2,c=0):
            return (b1-b2 + (w1[0]-w2[0])*x + c)/(w2[1]-w1[1])

        arg_w1 = model[1:,0]
        arg_w2 = model[1:,1]
        arg_b1 = model[0,0]
        arg_b2 = model[0,1]
        a0 = x1_min; a1 = f2(a0,arg_w1,arg_w2,arg_b1,arg_b2)
        b0 = x1_max; b1 = f2(b0,arg_w1,arg_w2,arg_b1,arg_b2)
        plt.plot([a0,b0], [a1,b1], "k")
        
    plt.show()

def plot_contour(svm,X,y,pos = 1,neg = -1):
    X_pos = X[y==pos]
    X_neg = X[y==neg]
    plt.plot(X_pos[:,0],X_pos[:,1],"r+",markersize = 3)
    plt.plot(X_neg[:,0],X_neg[:,1],"bo",markersize = 3)

    X1_plot = np.linspace(np.min(X[:,0]),np.max(X[:,0]),100)
    X2_plot = np.linspace(np.min(X[:,1]),np.max(X[:,1]),100)
    X1, X2 = np.meshgrid(X1_plot,X2_plot)
    X_new = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    y_new = svm.predict(X_new).reshape(X1.shape)

    plt.contour(X1, X2, y_new,[0.0], colors='k', linewidths=1)
    plt.axis("tight")
    plt.show()