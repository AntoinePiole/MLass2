from numpy import *
# Logistic Regression
import scipy.optimize as op
from logisticRegressionPredict import logisticRegressionPredict 
from logisticRegressionComputeGrad import logisticRegressionComputeGrad
from logisticRegressionComputeCost import logisticRegressionComputeCost
from kNNPredict import kNNPredict

def classify(trainSet, trainLabels, testSet):
	
	# Apply all methods 1 by 1
	
	## Logistic regression
	
	# Initialize fitting parameters
	initial_theta = zeros((3,1))
	
	# Run minimize() to obtain the optimal theta
	print('############ LOGISTIC REGRESSION ##############')
	print('Optimizing to obtain theta')
	Result = op.minimize(fun = logisticRegressionComputeCost, x0 = initial_theta, args = (trainSet, trainLabsel), method = 'TNC',jac = computeGrad);
	theta = Result.x;
	
	# Predict labels on test data
	predictedLabels = zeros(testSet.shape[0])
	predictedLabels = logisticRegressionPredict(array(theta), testSet)
	return predictedLabels
	
	## KNN
	
	# Set k
	# TODO : change k value
	k=3
	
	# Predict labels on test data
	predictedLabels = zeros(testSet.shape[0])
	for i in range(testSet.shape[0]):
		print("    Current Test Instance: " + str(i+1), " of ", I)
		predictedLabels[i] = kNNPredict(k, trainSet, trainLabels, testSet)
	return predictedLabels

	## AdaBoost
	
	## Decision Trees
	
	## SVM
	
	predictedLabels = zeros(testSet.shape[0])
	
	for i in range(testSet.shape[0]):
		if testSet[i,3] == 'female':
			predictedLabels[i] = 1

	return predictedLabels