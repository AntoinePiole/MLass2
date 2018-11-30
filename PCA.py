import numpy as np
import pandas
#my_data = pandas.read_csv('C:/Users/NSH7/Desktop/A FAIIIIIIIIIIIRE/ML/Lab Sessions/PCA/train.csv', delimiter=',')
#print(my_data)
# Data is a n*12 matrix filled with strings
#After preprocessing, we have a n*13 matrix
KJ=eye(13,13)
## 
def PCA(data,k):
    '''Performs principal components analysis (PCA) to data, a n*d matrix. We reduce the components to k.'''
    Mean_data = np.mean(data,0) #We compute the mean, data is a numpy array
    C = data - Mean_data # We subtract the mean (along columns) to data matrix
    W = np.dot(C.T, C) # compute covariance matrix
    eigval,eigvec = np.linalg.eig(W) # compute eigenvalues and eigenvectors of covariance matrix
    idx = eigval.argsort()[::-1] # Sort eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalue
    data_PCA = np.dot(C,real(eigvec[:,:k])) # Project the data to the new space (k dimension)
    return data_PCA

K=PCA(KJ,4)
print(K)
