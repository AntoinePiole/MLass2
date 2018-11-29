from numpy import *
from sigmoid import sigmoid

def logisticRegressionPredict(theta, X):
	# Predict whether the label is 0 or 1 using learned logistic 
	# regression parameters theta. The threshold is set at 0.5
	
	m = X.shape[0] # number of training examples
	n = X.shape[1] # each x is of size n
	
	c = zeros(m) # predicted classes of training examples
	
	p = zeros(m) # logistic regression outputs of training examples
	
	
	# ====================== YOUR CODE HERE ======================
	# Instructions: Predict the label of each instance of the
	#				training set.
	

		
		
	# evaluate part 1 : predict on X
	
	c = [0 for i in range(lenEval)]
	
	for j in range(lenEval):
		z = 0
		for k in range(n):
			z += theta[k] * Xeval[j][k]
		if logisticRegressionSigmoid(-z)>=0.5:
			c[j] = 1
		else :
			c[j] = 0
		

	# =============================================================
	
	return c

