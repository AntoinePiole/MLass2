from numpy import *
from math import e
from math import pow

def logisticRegressionSigmoid(z):
	# Computes the sigmoid of z.

	# ====================== YOUR CODE HERE ======================
	# Instructions: Implement the sigmoid function as given in the
	# assignment.
	g = 1/(1+exp(-z))

	# =============================================================
	return g
