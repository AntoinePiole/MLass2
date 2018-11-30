from numpy import *
from kNNEuclideanDistance import kNNEuclideanDistance

def getMajority(list):
    myMap = {}
    majorityElement = ( '', 0 ) # (occurring element, occurrences)
    for n in list:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1
        # Keep track of majorityElement on the go
        if (myMap[n] > majorityElement[1]) : 
            majorityElement = (n,myMap[n])

    return majorityElement

def kNNPredict(k, X, labels, y):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
    # training examples using the kNN with euclidean distance.
    #
    # Input: k: number of nearest neighbors
    #        X: training data           
    #        labels: class labels of training data
    #        y: test data
    
    
    # ====================== ADD YOUR CODE HERE =============================
    # Instructions: Run the kNN algorithm to predict the class of
    #               y. Rows of X correspond to observations, columns
    #               to features. The 'labels' vector contains the 
    #               class to which each observation of the training 
    #               data X belongs. and give y 
    #               the class of the majority of them.
    #
    # Note: To compute the distance betweet two vectors A and B use
    #       use the euclideanDistance(A,B) function.
    
    distances = []
    # Calculate the distance betweet y and each row of X
    for x in X :
        distances.append(kNNEuclideanDistance(x,y))
    # Find the k closest observations
    minIndexes = argpartition(asarray(distances), k)[:k]
    neighborLabels = labels[minIndexes]
    #print("neighborLabels : ", neighborLabels)
    label = getMajority(neighborLabels)[0]
    #print("Label is : ", label)

    # return the label of the test data    
    return label

 
