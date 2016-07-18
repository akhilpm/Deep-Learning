'''
CNN training and testing handler
'''

import os
import sys, getopt
import time
import numpy as np
import theano
import theano.tensor as T
from sklearn import preprocessing
from cnn import CNN
import pickle as cPickle
from logistic_sgd import LogisticRegression


def fit(data, labels, filename = 'weights.pkl'):
    fit_predict(data, labels, filename = filename, action = 'fit') 

def predict(test_dataset, filename = 'weights.pkl' ):
    return fit_predict(data=[], labels=[], filename= filename, test_datasets=[test_dataset], action = 'predict')[0] 


def fit_predict(data, labels, action, filename, test_datasets = [], learning_rate=0.1, n_epochs=20, nkerns=[20, 50], batch_size=500):
    seed = np.random.randint(1,10000)
    rng = np.random.RandomState(seed)
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    index = T.lscalar()  # index to a [mini]batch
    if action=='fit':
        NUM_TRAIN = len(data)
        if NUM_TRAIN % batch_size != 0: #if the last batch is not full, just don't use the remainder
            whole = (NUM_TRAIN / batch_size) * batch_size
            data = data[:whole]
            NUM_TRAIN = len(data) 

        # random permutation
        indices = rng.permutation(NUM_TRAIN)
        data, labels = data[indices, :], labels[indices]
        
        # batch_size == 500, splits (480, 20). We will use 96% of the data for training, and the rest to validate the NN while training
        is_train = np.array( ([0]* (batch_size - 20) + [1] * 20) * (NUM_TRAIN / batch_size))
        
        # now we split the dataset to test and valid datasets
        train_set_x, train_set_y = np.array(data[is_train==0]), labels[is_train==0]
        valid_set_x, valid_set_y = np.array(data[is_train==1]), labels[is_train==1]
        # compute number of minibatches 
        n_train_batches = len(train_set_y) / batch_size
        n_valid_batches = len(valid_set_y) / batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
        # allocate symbolic variables for the data
        epoch = T.scalar()
        #index = T.lscalar()  # index to a [mini]batch
        #x = T.matrix('x')  # the data is presented as rasterized images
        #y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        
        # construct the CNN class
        classifier = CNN(
            rng=rng,
            input=x,
            nkerns = nkerns,
            batch_size = batch_size
        )

        train_set_x = theano.shared(np.asarray(train_set_x, dtype=theano.config.floatX))
        train_set_y = T.cast(theano.shared(np.asarray(train_set_y, dtype=theano.config.floatX)), 'int32')  
        valid_set_x = theano.shared(np.asarray(valid_set_x, dtype=theano.config.floatX)) 
        valid_set_y = T.cast(theano.shared(np.asarray(valid_set_y, dtype=theano.config.floatX)), 'int32')
        
        validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        cost = classifier.layer3.negative_log_likelihood(y)
        # create a list of gradients for all model parameters
        grads = T.grad(cost, classifier.params)

        # specify how to update the parameters of the model as a list of (variable, update expression) pairs
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(classifier.params, grads)
        ]

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )




        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
        epoch = 0

        # here is an example how to print the current value of a Theano variable: print test_set_x.shape.eval()
        
        # start training
        while (epoch < n_epochs):
            epoch = epoch + 1   
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                '''
                if (epoch) % 5  == 0 and minibatch_index==0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                '''
        ###############
        # PREDICTIONS #
        ###############

        # save and load
        f = file(filename, 'wb')
        cPickle.dump(classifier.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        end_time = time.clock()              
        print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))


    if action == 'predict':
        # construct the CNN class
        classifier_2 = CNN(
            rng=rng,
            input=x,
            nkerns = nkerns,
            batch_size = batch_size
            )
        
        print "...."


        f = file(filename, 'rb')
        classifier_2.__setstate__(cPickle.load(f))
        f.close()


        RET = []
        for it in range(len(test_datasets)):
            test_data = test_datasets[it]
            N = len(test_data)
            test_data = theano.shared(np.asarray(test_data, dtype=theano.config.floatX))
            # just zeroes
            test_labels = T.cast(theano.shared(np.asarray(np.zeros(batch_size), dtype=theano.config.floatX)), 'int32')
        
            ppm = theano.function([index], classifier_2.layer3.pred_probs(),
                givens={
                    x: test_data[index * batch_size: (index + 1) * batch_size],
                    y: test_labels
                }, on_unused_input='warn')

            # p : predictions, we need to take argmax, p is 3-dim: (# loop iterations x batch_size x 2)
            p = [ppm(ii) for ii in xrange( N / batch_size)]  
            #p_one = sum(p, [])
            #print p
            p = np.array(p).reshape((N, 10))
            #print p
            p = np.argmax(p, axis=1)
            p = p.astype(int)
            RET.append(p)
        return RET


