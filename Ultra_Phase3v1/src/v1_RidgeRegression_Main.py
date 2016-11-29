

import cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
import pandas as pd
import theano
import theano.tensor as T
import copy

from v1_RidgeRegression_DataPrep import load_data


'''
#####################################################################################################

Ridge Regression Class

#####################################################################################################
'''

class CL_RidgeRegression(object):
    '''
    '''

    def __init__(self, input, n_in, n_out, Ridge_lambda):
        '''
        Initialize the parameters of the logistic regression

        input: theano.tensor.TensorType
               variable as one minibatch
               n_sample * n_in 
        n_in: the dimension of the space in which the datapoints lie
        n_out: int, here 2 or 3
               the dimension of the space in which the labels lie
        '''

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        # each row of W gives the n_out sentiment scores of corresponding hash tag from X_vector
        self.W = theano.shared( 
            value=numpy.zeros((n_in, n_out),dtype=theano.config.floatX),
            name='W',
            borrow=True)
        # initialize the biases b as a 1-d vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros((n_out,),dtype=theano.config.floatX),
            name='b',
            borrow=True)

        # probability distributions
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) )

        # this is effectively a classification problem
        # using Ridge Regression
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        self.Ridge_lambda = Ridge_lambda

        # keep track of model input
        self.input = input

    # neg_log as classification problem
    def negative_log_likelihood(self, y):
        '''
        y: theano.tensor.TensorType
           corresponds to a vector that gives for each example the correct label
        '''
        # indexing
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])+\
                self.Ridge_lambda*T.sum(self.W**2,axis=None)#+\
                #self.Ridge_lambda*T.sum(self.b**2,axis=None)

    # model errors, as percentage of incorrect predictions
    def errors(self, y):
        '''
        Return a float representing the percentage of errors of the minibatch

        y: theano.tensor.TensorType
           corresponds to a vector that gives for each example the correct label
        '''

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError( 'y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type)
                           )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

'''
#####################################################################################################

Main Function for Training

#####################################################################################################
'''

def RidgeRegression_Train(dataset_name, path_dataset, 
                          n_out=2, n_in=300,
                          learning_rate=0.01, n_epochs=1000, 
                          patience=10000, patience_increase=2, improvement_threshold=0.995,
                          Ridge_lambda=0.02,
                          batch_size=50, 
                          target_header=['spaceholder'], target_index=[0]):
    '''
    Main Function for Training Model
    Using simple Stochastic Gradient Descent

    dataset_name: e.g. 'perDayTweets_2016_11_03_train.pkl'
    path_dataset: e.g. '../Data/DataSet_Training/'
    
    learning_rate: earning rate used (factor for the stochastic gradient)
    n_epochs: maximal number of epochs to run the optimizer
    
    # early-stopping parameters #
    patience: look as this many minibatches regardless
    patience_increase: wait this much longer when a new best is found
    improvement_threshold: a relative improvement of this much is considered significant

    Ridge_lambda: weight of Ridge term on neg_log losses
    
    target_header: list of tags to extract
    target_index: list of tags' index
    '''
    assert target_header is not None
    assert target_index is not None

    #############
    # Load Data #
    #############

    datasets = load_data(dataset_name=dataset_name, path_dataset=path_dataset)
    # Note: data type of train_set_x here is GPU array type & theano.shared
    # rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # minibatch index
    index = T.lscalar() 
    # model symbolic variable
    x = T.matrix('x')  
    y = T.ivector('y')  

    # Construct the Ridge Regression Class
    classifier = CL_RidgeRegression(input=x, n_in=n_in, n_out=n_out, Ridge_lambda=Ridge_lambda)

    # the cost using neg_log as pure classification problem
    cost = classifier.negative_log_likelihood(y)

    # returns the precentage error of current model
    # given the X,Y data set
    test_model = theano.function( inputs=[index],
                                  outputs=classifier.errors(y),
                                  givens={ x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                           y: test_set_y[index * batch_size: (index + 1) * batch_size]
                                         }
                                )

    validate_model = theano.function( inputs=[index],
                                      outputs=classifier.errors(y),
                                      givens={ x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                               y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                             }
                                    )

    # compute the gradients
    g_W = T.grad(cost=cost, wrt=classifier.W)
    # update W 
    updates = [(classifier.W, classifier.W - learning_rate * g_W)]

    # training function
    # returns cost, with gradient updates
    # one cycle through fitting
    train_model = theano.function( inputs=[index],
                                   outputs=cost,
                                   updates=updates,
                                   givens={ x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                            y: train_set_y[index * batch_size: (index + 1) * batch_size]
                                          }
                                 )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'

    # go through this many minibatche before checking model on the validation set
    # in this case we check every epoch
    validation_frequency = n_train_batches
    # initialize validation_loss
    best_validation_loss = numpy.inf
    # define test_score
    test_score = 0.
    
    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0
    # cycle through n_epochs
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        # go through minibatches of this epoch
        for minibatch_index in range(n_train_batches):
            # one cycle through fitting
            minibatch_avg_cost = train_model(minibatch_index)
            
            ##############
            # Validation #
            ##############

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # get percentage errors from all minibatches of the validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                # mean of validation set percentage errors
                this_validation_loss = numpy.mean(validation_losses)
                #print "epoch %i, minibatch %i/%i, validation error %f %%" % \
                #       ( epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100. )

                # parameter display
                W_np = numpy.asarray( classifier.W.eval() )
                b_np = numpy.asarray( classifier.b.eval() )
                #print W_np[:10,:] 
                #print b_np
                
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    # improvement_threshold < 1 as 0.995, thus threshold per at least 0.5% improvement
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    # update best_validation_loss
                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    #print 'New Best Validation Loss \n epoch %i, minibatch %i/%i, test error of best model %f %%' % \
                    #      ( epoch, minibatch_index + 1, n_train_batches, test_score * 100. )

                    ###########################
                    # Saving Model Parameters #
                    ###########################
                    
                    # dataset_name: 'perDayTweets_2016_11_03_train.pkl'
                    file_name = 'MP'+dataset_name[-21:-10]+'.pkl'
                    with open(file_name, 'wb') as f:
                        pickle.dump(classifier, f)

            # End of if (iter + 1) % validation_frequency == 0:
            if patience <= iter:
                done_looping = True
                break
    # End of while (epoch < n_epochs) and (not done_looping):
    end_time = timeit.default_timer()
    
    print 'current Ridge lambda value: %f ' % Ridge_lambda
    print 'Optimization complete with best validation score of %f %%, with test performance %f %%' % \
            (best_validation_loss * 100., test_score * 100.)

    ###########################
    # Saving Model Parameters #
    ###########################

    W_np = numpy.asarray( classifier.W.eval() )
    W_list = [ [ W_np[idx,0], W_np[idx,1] ] for idx in range(len(W_np)) ]

    # output csv of W only
    csv_fileName = 'W'+dataset_name[-21:-9]+str(Ridge_lambda)+'.csv'

    OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
    data_file_name =  '../Outputs/' + csv_fileName
    Outputfilename = os.path.join(OutputfileDir, data_file_name) # ../ get back to upper level
    Outputfilename = os.path.abspath(os.path.realpath(Outputfilename))
    
    W_pd = pd.DataFrame(W_list)
    W_pd.to_csv(Outputfilename, index=False, header=False)
    print "Done saving Results into %s" % csv_fileName

    ############################################################################

    # output as b_np and indexed W_np 
    return (b_np, W_np[target_index,:], best_validation_loss * 100., test_score * 100.)


'''
#####################################################################################################

Execution 

#####################################################################################################
'''

if __name__ == '__main__':
    
    #########################################################################

    # dataset_name = 'perDayTweets_2016_11_09_train.pkl'
    # path_dataset = '../Data/DataSet_Training/'

    # Overall Tag List
    TagSelect_headerList_fig1 = ['trump','maga','hillary','imwithher','election2016','nevertrump','neverhillary',\
                                 'electionday', 'electionnight', 'fbi', 'notmypresident']
    TagSelect_indexList_fig1 = [0, 2, 10, 3, 4, 8, 14,\
                                5, 1, 40, 21]

    #########################################################################

    flag_Ridge_list = True 

    dataset_name = 'perDayTweets_2016_10_31_train.pkl'
    path_dataset = '../Data/DataSet_Training/'

    # create Ridge parameter list
    Ridge_lambda_list = [0]*11
    for idx in range(11):
        Ridge_lambda_list[idx] = idx*0.001
    for idx in range(8):
        Ridge_lambda_list.append( 0.01+0.005*(idx+1) )
    print "Ridge_lambda_list"
    print Ridge_lambda_list

    #########################################################################
    
    if flag_Ridge_list == True:
        
        FinalResult_list = []
        header = ['RL', 'Er_valid', 'Er_test']
        header = header + TagSelect_headerList_fig1
        FinalResult_list.append(header)

        for Ridge_val in Ridge_lambda_list:
            Returns = RidgeRegression_Train(dataset_name=dataset_name, path_dataset=path_dataset, 
                                            n_out=2, n_in=300,
                                            learning_rate=0.01, n_epochs=1000, 
                                            patience=25000, patience_increase=2, improvement_threshold=0.995,
                                            Ridge_lambda=Ridge_val,
                                            batch_size=50,
                                            target_header=TagSelect_headerList_fig1, 
                                            target_index=TagSelect_indexList_fig1)
            # extract (b_np, W_np[target_index,:])
            b_vector = Returns[0]
            W_vector_list = Returns[1]
            valid_Er_pct = Returns[2]
            test_Er_pct = Returns[3]
            # W_2_list
            W_2_list = [Ridge_val, valid_Er_pct, test_Er_pct]
            W_2_list = W_2_list + [ W_vector_list[idx,1] for idx in range(len(W_vector_list)) ]
            FinalResult_list.append(W_2_list)

        # output to .csv
        csv_fileName = 'ScanRidgeValue_woBias_'+dataset_name[-21:-10]+'.csv'

        OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
        data_file_name =  '../Outputs/' + csv_fileName
        Outputfilename = os.path.join(OutputfileDir, data_file_name) # ../ get back to upper level
        Outputfilename = os.path.abspath(os.path.realpath(Outputfilename))
        
        FinalResult_pd = pd.DataFrame(FinalResult_list)
        FinalResult_pd.to_csv(Outputfilename, index=False, header=False)
        print "Done saving Results into %s" % csv_fileName


