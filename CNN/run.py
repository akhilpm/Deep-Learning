'''
CNN main file
'''

import os
import sys, getopt
import time
import numpy as np
from cnn_training_computation import fit, predict
from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_lfw_people

def run():
    # read the data, labels
    mnist = fetch_mldata('MNIST original')
    mnist.data = mnist.data/255.0
    mnist.target = mnist.target.astype(np.int32)
    trainX, testX, trainY, testY = train_test_split(mnist.data, mnist.target, test_size = 0.3, random_state=42)
   
    trainX = trainX[0:2000]
    testX = testX[0:1000]
    trainY = trainY[0:2000]
    testY = testY[0:1000]


    """
    # TO USE FACE DATASET, UNCOMMENT THIS PART
    # NOTE: the lfw_people.target variable should contain the age-group labels 

    lfw_people = fetch_lfw_people()
    lfw_people.target = label(age_groups)
    trainX, testX, trainY, testY = train_test_split(lfw_people.data, lfw_people.target, test_size = 0.3, random_state=42)
    trainX = trainX[0:2000]
    testX = testX[0:1000]
    trainY = trainY[0:2000]
    testY = testY[0:1000]    
    """

    start = time.time()
    # DO argmax
    #labels = np.argmax(labels, axis=1)
    #print labels
        
    # normalization
    amean = np.mean(trainX)
    data = trainX - amean
    astd = np.std(trainX)
    trainX = trainX / astd
    # normalise using coefficients from training data
    testX = (testX - amean) / astd
    #valid_data = (valid_data - amean) / astd
    print('Data pre-processing time : %f Minutes\n' %((time.time()-start)/60))

    start = time.time()
    fit(trainX, trainY)
    print('Train Time : %f Minutes\n' %((time.time()-start)/60))

    start = time.time()
    pred = predict(testX)
    print('Accuracy on test set : %f\n' %(accuracy_score(testY, pred)*100))

    print('Test Time : %f Minutes\n' %((time.time()-start)/60))


    '''
    print "finished training"
    rv = predict(valid_data)
    rt = predict(test_data)



    # UNDO argmax and save results x 2
    r = rv
    N = len(r)
    res = np.zeros((N, 10))
    for i in range(N):
        res[i][r[i]] = 1
    
    np.savetxt("mnist_valid.predict", res, fmt='%i')
    
    r = rt
    N = len(r)
    res = np.zeros((N, 10))
    for i in range(N):
        res[i][r[i]] = 1
    
    np.savetxt("mnist_test.predict", res, fmt='%i')
    print "finished predicting."
    '''   


if __name__ == '__main__':
    run()

