

from __future__ import print_function

import json
import sys
import time
import os
import numpy
import pandas as pd
from collections import OrderedDict
import csv
import itertools

import nltk
import six.moves.cPickle as pickle

import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from v1_LSTM_Utilities import *
import v1_LSTM_DataPrep 

from P2P2_Tags_Operations import Load_Predictions


"""
####################################################################################

Public Variables and Functions

####################################################################################
"""

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def _p(pp, name):
    '''
    giving parameter names, as theano variablename etc. 
    '''
    return '%s_%s' % (pp, name)


"""
####################################################################################

LSTM Functions

####################################################################################
"""

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    '''
    LSTM layer function
    performs calculations on the layer, per step-wise

    tparams: theano parameter set
             OrderedDict() with elements as theano.shared()

    state_below: as emb, post embedding x matrix

    options: hyper parameters; dict format; dict of all model hyper parameters
    prefix: 
    mask: theano tensor variable; Sequence mask; from imdb.prepare_data
          mask marked each sentence on the X matrix into the same length
    
    returns
    ----
    perform one iteration through X matrix, returns a list of hidden value generated
    '''
    
    # nsteps as unified length of sentences
    nsteps = state_below.shape[0] 
    # n_samples is the number of sentences in a minibatch
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    # important for representing sentences of different length loaded into X matrix
    assert mask is not None
    
    ####################################################################################
    # slice the concatenated W matrix
    def _slice(_x, n, dim):
        '''
        minibatch creator
        n: index of variable matrix
        dim: size of variable matrix
        '''
        # _x.ndim == 3
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        # _x.ndim == 2
        return _x[:, n * dim:(n + 1) * dim]
    # one step of the LSTM operation
    def _step(m_, x_, h_, c_):
        '''
        m_: mask
            as (n_timesteps, n_samples), 
            but after 'theano.scan', as (n_samples)
        x_: as W*Xt + b
            (n_timesteps, n_samples, options['dim_proj']), 
            but after 'theano.scan', as (n_samples, options['dim_proj'])
        h_: hidden state vector; (n_samples, options['dim_proj'])
        c_: past state of LSTM cell; (n_samples, options['dim_proj'])
        '''
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        # i, f, o, c are all concantinated together
        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c 

        h = o * tensor.tanh(c)
        h = m_[:, None] * h

        return h, c
    
    ####################################################################################    
    
    # W*Xt + b
    state_below = ( tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                    tparams[_p(prefix, 'b')] )

    ####################################################################################    

    # theano.scan() go through sequences=[mask, state_below]
    # performing one iteration through the data
    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

    return rval[0]


'''
####################################################################################
'''
# cost reduce algorithms

def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.
    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    # theano updates tuples
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

'''
####################################################################################
'''

def build_model(tparams, options):
    """
    LSTM model building function

    Parameters
    ----------
    tparams: theano.shared() variable dict
    options: dict() of parameters
    """
    # SEED, universal variable
    trng = RandomStreams(SEED)
    # Used for dropout. This shared value will be modified during training!
    use_noise = theano.shared(numpy_floatX(0.))
    # theano variable
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')
    # theano varialbe based on variable X; columns of X are sentences
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    
    ####################################################################################
    
    ################################
    # Embedding and Context Window #
    ################################
    
    # LSTM matrix_embed
    emb = tparams['Wemb'][x.flatten()].reshape([ n_timesteps, n_samples, options['dim_proj'] ])

    ####################################################################################

    # LSTM Layer
    proj = lstm_layer(tparams, emb, options, prefix=options['encoder'], mask=mask)

    ##################################################  
    # How Hidden Values from LSTM Layer is Processed #
    ##################################################  

    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0) 
        # numerical average of hidden values alone each sentences
        proj = proj / mask.sum(axis=0)[:, None]
     
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    ################################################### 
    # How Hidden Values are used to create Prediction #
    ################################################### 
    
    pred = tensor.nnet.softmax( tensor.dot(proj, tparams['U']) + tparams['b'] )

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log( pred[ tensor.arange(n_samples), y ] + off ).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


'''
####################################################################################
'''

def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ 
    If you want to use a trained model, this is useful to compute
    the probabilities of new examples.

    parameters:
    ---
    f_pred_prob:
    prepare_data:
    data:
    iterator:

    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs

def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error

    parameters:
    ---
    f_pred: Theano function computing the prediction
    prepare_data: prepare_data for that dataset. IS a function
    data:
    iterator:
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


"""
####################################################################################

Main Function

####################################################################################
"""

def train_lstm(
    dim_proj=256,  
    # word embeding dimension and LSTM number of hidden units.
    n_words=1000,  
    # Vocabulary size
    
    patience=5,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    
    optimizer='adadelta',  
    # sgd, adadelta and rmsprop available, 
    # sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    # optimizer's value is a function
    encoder='lstm',  # TODO: can be removed must be lstm.
    
    # The best model will be saved there    
    saveto='lstm_model.npz',  
    # load parameters from .npz files
    loadfrom = 'lstm_model.npz',
    # name of the data set; also name of the .pkl file
    dataset='tweetText_tagScore.pkl',
    # path to the data set
    dataset_path="../Data/DataSet_Tokenize",

    validFreq=100,  # Compute the validation error after this number of update.
    saveFreq=100,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=64,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    
    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worse test error
                       # This frequently need a bigger model.
    reload_model=False,  
    test_size=-1  # If >0, we keep only this number of test example.
):

    model_options = locals().copy()
    print("Initial model options", model_options)

    load_data =  v1_LSTM_DataPrep.load_data
    prepare_data = v1_LSTM_DataPrep.prepare_data

    print('Loading data')
    train, valid, test = load_data(dataset_name=dataset, path_dataset=dataset_path, 
                                   n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    if test_size > 0:
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

    # saving model_options to .json file
    json_name = saveto[:-4] + '.json'
    with open(json_name, 'w') as fp:
        json.dump(model_options, fp, sort_keys=True, indent=4)

    ####################################################################################

    ##################
    # Building Model #
    ##################
    
    print('Building model')

    params = init_params(model_options)
    # loading previous parameters or not
    if reload_model == True:
        # path = name.npz, in the save address as this code
        params = load_params(path=loadfrom, params=params)
        json_name = saveto[:-4] + '_PredReload.json'
        with open(json_name, 'w') as fp:
            json.dump(model_options, fp, sort_keys=True, indent=4)

    tparams = init_tparams(params)
    
    # build_model()
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    # Weight decay for the classifier applied to the U weights.
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # cost, grads, lr and grad_updates
    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    if optimizer == 'adadelta':
        optimizer=adadelta
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)

    ##################
    # Training Model #
    ##################
    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0
    best_p_str = ''
    flag_eidx = -1

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                use_noise.set_value(1.)

                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)
                # check cost for nan and inf
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                # display
                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                # validation
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)
                    print( '\nTrain ', train_err, 'Valid ', valid_err, 'Test ', test_err, 'history_errs ', len(history_errs)+1, '\n' )

                    history_errs.append([valid_err, test_err])                      
                    
                    if (best_p is None) or (valid_err<numpy.array(history_errs)[:-1,0].min()) or (test_err<numpy.array(history_errs)[:-1,1].min()):
                        # renew best_p
                        best_p = unzip(tparams)
                        
                        # saving best_p to 'saveto'
                        print('Saving...')
                        params = best_p

                        numpy.savez(saveto, history_errs=history_errs, **params)

                        pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                        print('Done')
                        
                        # reset bad_counter
                        bad_counter = 0
                        print('best_p updated; bad_counter reset to 0;')
                        best_p_str = "best_p Train: %f, Valid: %f, Test: %f" % tuple( [train_err]+[valid_err]+[test_err] )
                        print(best_p_str)               

                    if (eidx>3) and (valid_err>=numpy.array(history_errs)[:-1,0].min()) and (test_err>=numpy.array(history_errs)[:-1,1].min()):
                        bad_counter += 1
                        print('bad_counter: ', bad_counter, 'patience: ', patience)                 
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print( best_p_str )
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)

    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    
    ####################################################################################
    print("Final model options", model_options) 
    return params



"""
####################################################################################

Prediction Code

####################################################################################
"""

def load_predict_lstm(dataset, dataset_path, 
                      loadfrom, 
                      MySQL_DBkey,
                      pred_columnName, sql_tableName, 
                      params_direct=None,
                      model_options=None):
    '''
    dataset: name of the data set; also name of the .pkl file
    dataset_path: path to .pkl files
                e.g. dataset_path="../Data/DataSet_Tokenize"

    pred_columnName: column name for prediction results on filtered_tweet_stack
    sql_tableName: full table name of filtered_tweet_stack_'sth'

    '''
    
    ###############################
    # load Model hyper-parameters #
    ###############################
    
    reload_model = True # naturally

    if reload_model == True :

        model_options_fileName = loadfrom[:-4] + '.json'
        with open(model_options_fileName, 'r') as fpr:
            model_options = json.load(fpr)  

        json_name = loadfrom[:-4] + '_PredReload.json'
        with open(json_name, 'w') as fpl:
            json.dump(model_options, fpl, sort_keys=True, indent=4)

        params = init_params(model_options)
        params = load_params(path=loadfrom, params=params)

    if params_direct is not None:
        params = params_direct
        print("params set to params_direct")

    n_words = model_options['n_words']
    maxlen = model_options['maxlen']

    print("model parameter check")
    print("N_words: ", n_words, "dim_proj: ", model_options['dim_proj'], "maxlen: ", maxlen)

    dataset = dataset
    dataset_path = dataset_path

    batch_size = model_options['batch_size']

    ####################################################################################
    
    #############
    # load Data #
    #############

    # data prep functions
    load_data = v1_LSTM_DataPrep.load_data_for_prediction
    prepare_data = v1_LSTM_DataPrep.prepare_data

    print('Loading data')
    dataset4prediction, fileNames = load_data(dataset_name=dataset, path_dataset=dataset_path, 
                                              n_words=n_words, maxlen=maxlen)

    scores_tag = dataset4prediction[1]

    ydim = numpy.max(dataset4prediction[1]) + 1
    if model_options['ydim'] != ydim:
        print( "Error: prediction data set ydim NOT MATCHING training data set ydim" )
        print( "training data set ydim: %i" % model_options['ydim'] )
        print( "prediction data set ydim: %i" % ydim )
        return None

    ##################
    # Building Model #
    ##################
    print('Building model')

    tparams = init_tparams(params)
    
    (use_noise, x, mask, y, 
        f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    ######################
    # Predicting Results #
    ######################
    print('Making Predictions')
    print("%d prediction examples" % len( dataset4prediction[0] ) )

    dataset_sentences_digit = []
    dataset_preds_prob = []
    dataset_preds = []

    try:
        pred_index_batches = get_minibatches_idx(len(dataset4prediction[0]), batch_size, 
                                                 shuffle=False)
        n_samples = 0

        for _, pred_index in pred_index_batches:
            
            ########################
            # preds and preds_prob #
            ########################
            
            # Select the examples for this minibatch
            y = [ dataset4prediction[1][t] for t in pred_index ]
            x = [ dataset4prediction[0][t] for t in pred_index ]

            x, mask, y = prepare_data(x, y)
            n_samples += x.shape[1]

            print('Seen %d samples' % n_samples)            
            
            preds_prob = f_pred_prob(x, mask)
            preds = f_pred(x, mask)

            ##############################
            # save to np.array as digits #
            ##############################

            # convert to numpy.array
            X_values = numpy.asarray(x)
            preds_prob = numpy.asarray(preds_prob)
            preds = numpy.asarray(preds)
            # rotate axis
            X_values = numpy.swapaxes(X_values, 0, 1)

            # load into dataset_preds_prob and dataset_preds
            if len(dataset_preds_prob) == 0 or len(dataset_preds) == 0:
                # initialized
                dataset_preds_prob = preds_prob
                dataset_preds = preds
            else:
                dataset_preds_prob = numpy.concatenate( (dataset_preds_prob, preds_prob), 
                                                  axis=0)
                dataset_preds = numpy.concatenate( (dataset_preds, preds), 
                                                  axis=0)
    except KeyboardInterrupt:
        print("Prediction interupted")
        return None

    ####################################################################################
    
    #####################################
    # Read, Expand and Load MySQL table #
    #####################################
    
    # tuple with lists or numpy.arrays as elements
    print("\n\nlengthes: ", len(fileNames), len(scores_tag), len(dataset_preds_prob), len(dataset_preds) )
    fileName_Scores_tuple = ( fileNames, scores_tag )
    print("Start Loading %s into %s" % tuple( [pred_columnName]+[sql_tableName])
         )
    predictions_tuple = ( dataset_preds_prob, dataset_preds)

    Load_Predictions(MySQL_DBkey=MySQL_DBkey,
                     pred_columnName=pred_columnName, sql_tableName=sql_tableName, 
                     fileName_Scores_tuple=fileName_Scores_tuple, predictions_tuple=predictions_tuple
                     )

    ####################################################################################
    return None


"""
####################################################################################

Test Codes

####################################################################################
"""

if __name__ == '__main__':
    
    ####################################################################

    MySQL_DBkey = {'host':'localhost', 'user':'', 'password':'', 'db':'','charset':'utf8'}

    ####################################################################################
    dict_DataSet_Full = {
        'dataset':'FullScoredTweets_train', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Training/',
        # same for all cases
        'lstm_saveto': 'lstm_DataSetFull',
        'lstm_loadfrom':'lstm_DataSetFull',
        }
    
    ####################################################################
    # training
    flag_training = False    
    dict_training = dict_DataSet_Full

    ####################################################################
    # predicting
    flag_predicting = True   

    pred_columnName = 'N30KH512'
    sql_tableName = 'prediction_fullscoredtweets'

    path_Predicting = '../Data/DataSet_Predicting/'

    dict_DataSet_ = {
        'dataset':'FullScoredTweets_PredictCheck', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Predicting/',
        # same for all cases
        'lstm_saveto': 'lstm_DataSetFull_Nwords_30000_dimProj_512',
        'lstm_loadfrom':'lstm_DataSetFull_Nwords_30000_dimProj_512',
        }
    dict_predicting = dict_DataSet_
    
    ####################################################################
    # for training !!!
    if flag_training == True:

        para_dataset = dict_training['dataset'] + '.pkl'
        para_dataset_path = dict_training['dataset_path']

        para_n_words = 30000
        dim_proj = 512

        para_saveto = dict_training['lstm_saveto']
        para_saveto = para_saveto + '_Nwords_' + str(para_n_words)
        para_saveto = para_saveto + '_dimProj_' + str(dim_proj)
        para_saveto = para_saveto + '.npz'

        para_loadfrom = para_saveto

        params = train_lstm(
            max_epochs=100,
            patience=10, # Number of validation period to wait before early stop if no progress
            saveFreq=500,
            validFreq=500,
            test_size=-1, 
            # If >0, we keep only this number of test example.
            
            dim_proj=dim_proj, # word embeding dimension and LSTM number of hidden units.
            n_words=para_n_words, # Vocabulary size
            
            dataset=para_dataset,
            # name of the data set, 'sth.pkl'; also name of the .pkl file
            # datasets = {'Name?': (LSTM_DataPrep.load_data, LSTM_DataPrep.prepare_data)}
            dataset_path=para_dataset_path,
            # path to the data set

            saveto=para_saveto,
            loadfrom = para_loadfrom,
            reload_model=False  # whether reload revious parameter or not
            )

    ####################################################################
    # for predicting !!!
    if flag_predicting == True:

        if flag_training == False:
            params = None
        
        if params is not None:
            print("\n\n using training params directly \n\n")

        ####################################################################
        # setting path for .txt files
        dict_predicting['dataset_path'] = path_Predicting

        ####################################################################
        # parameters of train_lstm()
        para_dataset = dict_predicting['dataset'] + '.pkl'
        para_dataset_path = dict_predicting['dataset_path']
        
        para_loadfrom = dict_predicting['lstm_loadfrom']
        para_loadfrom = para_loadfrom + '.npz' 

        ####################################################################
        load_predict_lstm(dataset=para_dataset, 
                          dataset_path=para_dataset_path, 
                          loadfrom=para_loadfrom,
                          MySQL_DBkey=MySQL_DBkey,
                          pred_columnName=pred_columnName, 
                          sql_tableName=sql_tableName, 
                          params_direct=params, 
                          model_options=None)

