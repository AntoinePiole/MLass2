import numpy as np
from sklearn.svm import SVC


def gaussianKernel(X1, X2, sigma = 0.1):
    
    m = X1.shape[0]
    K = np.zeros((m,X2.shape[0]))
    
    for i in range(m):
        K[i,:] = np.exp((-(np.linalg.norm(X1[i,:]-X2, axis=1)**2))/(2*sigma**2))
    
    return K

def SVM_gaussian(X1,y1,X2,C,sigma):
    svc = SVC()  #if nothing, use Radial basis function kernel, sigma=1
    svc.fit(X1,y1)
    return svc.predict(X2)


