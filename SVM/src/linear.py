import numpy as np

class LinearClassfication(object):
    def __init__(self,intercept = True,resturct = True,verbose = False,lamb = 1):
        self.intercept = intercept
        self.restruct = resturct
        self.verbose = verbose
        self.lamb = lamb

    def add_intercept(self,X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1) 

    def restruct_data(self,t,kind=2):
        y = np.zeros([t.size,kind])
        y[t == 1] = [1,0]
        y[t == -1] = [0,1]
        return y

    def cost(self,y,t):
        m = y.size
        t1 = np.square(y-t).reshape(m)
        [n,k] = self.w.shape
        t2 = np.square(self.w[1:,:].reshape((n-1)*k))
        return t1.mean()*2 + self.lamb/2.0 * np.sum(t2)/m
        
    def train(self,X,t):
        if self.intercept:
            X = self.add_intercept(X)
        if self.restruct:
            t = self.restruct_data(t)
        feature = X.shape[1]
        reg = np.eye(feature)
        self.w = np.dot(np.dot(np.linalg.inv(self.lamb * reg + np.dot(X.T,X)),X.T),t)
        y = np.dot(X,self.w)
        # t = temp
        if self.verbose:
            print "the cost of linear classfication is %f" % (self.cost(y,t))

    def predict(self,X):
        if self.intercept:
            X = self.add_intercept(X)
        y = np.dot(X,self.w)
        index = np.argmax(y,axis=1)
        t = -np.ones(X.shape[0])
        t[index == 0] = 1

        return t
