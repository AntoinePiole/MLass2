from numpy import *
from logisticRegressionSigmoid import logisticRegressionSigmoid

def computeGrad(theta, X, y):
	# print('computeGrad called')
	# Computes the gradient of the cost with respect to
	# the parameters.
	
	m = X.shape[0] # number of training examples
	n = X.shape[1]
	
	grad = zeros(size(theta)) # initialize gradient
	
	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the gradient of cost for each theta,
	# as described in the assignment.

	for j in range (size(theta)) :
		for i in range(m):
			z = 0
			for k in range(n):
				z += theta[k] * X[i][k]
			grad[j] += ( (logisticRegressionSigmoid(-z)) -y[i] ) * X[i][j]
		grad[j] *= -1/m
		
	# =============================================================
	
	return grad
