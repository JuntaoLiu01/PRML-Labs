import numpy as np

class LogisticRegression(object):
    def __init__(self,alpha=0.01,lamb = 1,num_iter=1000,intercept=True,verbose=False):
        self.alpha = alpha
        self.lamb = lamb
        self.num_iter = num_iter
        self.intercept = intercept
        self.verbose = verbose

    def add_intercept(self,X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1) 

    def sigmoid(self,z):
        return 1/(1.0+np.exp(-z))
   
    def cost(self,h,t):
        m = t.size
        t1 = -np.multiply(t,np.log(h)) - np.multiply(1-t,np.log(1-h))
        t2 = self.lamb/2.0 * np.square(self.w[1:])
        return t1.mean() + np.sum(t2)/m

    def train(self,X,t):
        if self.intercept:
            X = self.add_intercept(X)
        m = t.size
        t[t==-1] = 0
        self.w = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X,self.w)
            h = self.sigmoid(z)

            if self.verbose:
                print "%dth iteration,cost is %f" % (i+1,self.cost(h,t)) 

            gradient = np.dot(X.T,h-t)
            temp = gradient[0]
            gradient += self.lamb * self.w
            gradient[0] = temp
            self.w -= self.alpha/m * gradient 
        t[t==0] = -1

    def predict(self,X,threshold):
        if self.intercept:
            X = self.add_intercept(X)
        z = np.dot(X,self.w)
        h = self.sigmoid(z)
        h[h >= threshold] = 1
        h[h < threshold] = -1
        return h

    # def cost(self,h):
    #     return -np.log(h).mean() + self.lamb/2 * np.square(self.w[1:]).mean()

    # def train(self,X,t):
    #     if self.intercept:
    #         X = self.add_intercept(X)

    #     m = t.size
    #     # weights initialization
    #     self.w = np.zeros(X.shape[1])

    #     for i in range(self.num_iter):
    #         z = np.dot(X, self.w)
    #         h = self.sigmoid(np.multiply(t,z))
    #         if self.verbose == True:
    #             print "%dth iteration,cost is %f" % (i+1,self.cost(h))

    #         gradient = np.dot(X.T,-np.multiply((1-h),t))
    #         temp = gradient[0]
    #         gradient += self.lamb * self.w
    #         gradient[0] = temp
    #         self.w -= self.alpha/m * gradient 
    #         # print self.w

    # def predict_prob(self,X,t):
    #     if self.intercept:
    #         X = self.add_intercept(X)
    #     z = np.dot(X,self.w)
    #     return self.sigmoid(np.multiply(t,z))
    
    # def predict(self, X,t,threshold):
    #     res = -np.ones(t.shape)
    #     res[self.predict_prob(X,t) >= threshold] = 1
    #     return res