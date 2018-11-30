from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import classify
from preprocess import preprocess

# Load data
csv_file_object = csv.reader(open('train.csv', 'rt')) # Load in the csv fileù
header = csv_file_object.__next__() 					  # Skip the fist line as it is a header
data=[] 											  # Create a variable to hold the data

for row in csv_file_object: # Skip through each row in the csv file,
    data.append(row[0:]) 	# adding each row to the data variable
X = array(data) 		    # Then convert from a list to an array.
X = array([x[0].split(',') for x in X])

y = X[:,1].astype(int) # Save labels to y 

X = delete(X,1,1) # Remove survival column from matrix X
X = preprocess(X) # Turn X into a "normalized" float matrix, with 0s where data is missing
                  # Not really normalized, as it is normalized not taking missing values into account

# Initialize cross validation
kf = cross_validation.KFold(X.shape[0], n_folds=10)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  

for trainIndex, testIndex in kf:
    trainSet = X[trainIndex]
    testSet = X[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
    
    
    #Predict
    predictedLabels = classify(trainSet, trainLabels, testSet, "neuralNetwork")
    
    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print ('Accuracy: ' + str(float(correct)/(testLabels.size)))
    totalCorrect += correct
    totalInstances += testLabels.size
print ('Total Accuracy: ' + str(totalCorrect/float(totalInstances)))
    	
    
