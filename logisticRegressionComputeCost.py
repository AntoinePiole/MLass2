from numpy import *
from logisticRegressionSigmoid import logisticRegressionSigmoid

def logisticRegressionComputeCost(theta, X, y): 
	# Computes the cost using theta as the parameter 
	# for logistic regression. 
	
	# X = [xi] avec m xi
	# chaque xi est de longueur n
	# print("X is of type : ", type(X), " with shape : ", X.shape)
	# print("theta is of type : ", type(theta), " with shape : ", theta.shape)
	
	m = X.shape[0] # number of training examples : 
	n = X.shape[1]
	
	J = 0
	Jp = 0
	Jn = 0
	
	# ====================== YOUR CODE HERE ======================
	#               that is described by theta (see the assignment 
	#				for more details).

	for i in range(m):
		z = 0
		for j in range(n):
			z += theta[j]*X[i][j]
		Jp += y[i]*log(sigmoid(-z))
		Jn += (1-y[i])* log(1 - logisticRegressionSigmoid(-z))
	J = -1/m*(Jp+Jn)
	# =============================================================
	
	return J
