

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
import cPickle as pkl

import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


"""
####################################################################################

Public Variables and Functions

####################################################################################
"""

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

"""
####################################################################################

Auxiliary Functions

####################################################################################
"""

def numpy_floatX(data):
    '''
    return data in np.array format, dtype as theano float32
    '''
    return numpy.asarray(data, dtype=config.floatX)

def _p(pp, name):
    '''
    giving parameter names, as theano variablename etc. 
    '''
    return '%s_%s' % (pp, name)

def get_minibatches_idx(n, minibatch_size, shuffle=True):
    """
    Default to shuffle the dataset at each iteration.
    n: total number of lines/samples
    """
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = [] # list of list of indexes
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    
    # Make a minibatch out of what is left
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def dropout_layer(state_before, use_noise, trng):
    '''
    state_before: numerical average of hidden values alone each sentences
    use_noise: use_noise = theano.shared(numpy_floatX(0.))
               This shared value will be modified during training!
    trng: random number generator; trng = RandomStreams(SEED)
    '''
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)
                         ),
                         state_before * 0.5)
    return proj

"""
####################################################################################

Model Parameter Operations

####################################################################################
"""

def zipp(params, tparams):
    '''
    When we reload the model. Needed for the GPU stuff.
    tparams is the parameter set used by model
    '''
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

def unzip(zipped):
    '''
    When we pickle the model. Needed for the GPU stuff.
    return a new parameter set, values loaded from zipped
    '''
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def load_params(path, params):
    '''
    np.load(): Load arrays or pickled objects from .npy, .npz or pickled files.
    return parameter set, loaded from .npz or pickled files at path
    '''
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def init_tparams(params):
    '''
    initialize and return theano parameters, loaded from params
    ---
    params: coming from init_params()
    ---
    returns:
    ---
    tparams is OrderedDict() with elements as theano.shared()
    tparams is fed into build_model() then lstm_layer()
    '''
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    parameters initialized as numpy.arrays
    in def train_lstm():, parameters return by init_params() is fed into init_tparams()
    """
    params = OrderedDict()
    
    ####################################################################################
    # embedding
    # options['n_words']+1 is the minimum
    randn = numpy.random.rand(options['n_words']+1, 
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    ####################################################################################

    # LSTM layer, theano tensor variables
    params = param_init_lstm(options,  params, prefix=options['encoder'])
    
    ####################################################################################    
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim) 
    # random normal distribution, with mean 0 and variance 1
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter: as numpy.arrays

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1) # after concantenation, W has 3 dimentions
    params[_p(prefix, 'W')] = W
    
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params



