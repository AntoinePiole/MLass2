from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import classify
from preprocess import preprocess

from PCA import PCA 

## Load training data

csv_file_object = csv.reader(open('train.csv', 'rt')) # Load in the csv file
header = csv_file_object.__next__() 					  # Skip the fist line as it is a header
data=[] 											  # Create a variable to hold the data

for row in csv_file_object: # Skip through each row in the csv file,
    data.append(row[0:]) 	# adding each row to the data variable
X = array(data) 		    # Then convert from a list to an array.
X = array([x[0].split(',') for x in X])

y = X[:,1].astype(int) # Save labels to y 

X = delete(X,1,1) # Remove survival column from matrix X

## Load test data

csv_file_object = csv.reader(open('test.csv', 'rt')) # Load in the csv file
header = csv_file_object.__next__() 					  # Skip the fist line as it is a header
dataTest=[] 											  # Create a variable to hold the data

for row in csv_file_object: # Skip through each row in the csv file,s
    dataTest.append(row[0:]) 	# adding each row to the data variable
Xtest = array(dataTest) 		    # Then convert from a list to an array.
ids = Xtest[:,0].astype(int) # Save ids to ids 

titles = array([x.split(',')[1] for x in names]).transpose()
        
Xtest=insert(Xtest, 2, Xtest[:,2], 1)
for k in range(Xtest.shape[0]):
    title=(Xtest[k,2]).split(',')[1]
    Xtest[k,3]=title

Xall = np.concatenate((X,Xtest), axis=0)

Xall = preprocess(Xall) # Turn X into a "normalized" float matrix, with 0s where data is missing
                  # Not really normalized, as it is normalized not taking missing values into account

# Choosing classifier
classifier = "SVM"

if classifier == "logisticRegression" :
    Xall = PCA(Xall, 5)
    
elif classifier == "kNN" :
    Xall = PCA(Xall, 20)
# elif classifier == "adaBoost" : no PCA needed
elif classifier == "SVM" :
    Xall = PCA(Xall, 12)

# elif classifier == "neuralNetwork" : no PCA needed

X = Xall[0:891]
Xtest = Xall[892::]
## Initialize cross validation

kf = cross_validation.KFold(X.shape[0], n_folds=10)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  
'''
for trainIndex, testIndex in kf:
    trainSet = X[trainIndex]
    testSet = X[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
    
    
    #Predict

    predictedLabels = classify(trainSet, trainLabels, testSet, classifier)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print ('Accuracy: ' + str(float(correct)/(testLabels.size)))
    totalCorrect += correct
    totalInstances += testLabels.size
    
print ('Total Accuracy: ' + str(totalCorrect/float(totalInstances)))
'''


## Compute results

predictedLabels = classify(X, y, Xtest, classifier)

## Return results as CSV

with open('names.csv', 'w', newline='') as csvfile:
    fieldnames = ['PassengerId', 'Survived']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(Xtest.shape[0]):
        writer.writerow({'PassengerId' : ids[i], 'Survived' : predicteLabels[i]})


