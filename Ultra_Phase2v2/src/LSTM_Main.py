
from __future__ import print_function
import six.moves.cPickle as pickle


from collections import OrderedDict
import sys
import time
import numpy
import pandas as pd
import csv
import os
import json

import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# data preparation file
import LSTM_DataPrep

# predicting reloading
from MarkedTag_Import import Load_Predictions


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

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
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

    if (minibatch_start != n):
        # Make a minibatch out of what is left
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
    # 
    randn = numpy.random.rand(options['n_words'], 
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    ####################################################################################

    # See get_layer() below
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    # from def train_lstm():
    # ydim = numpy.max(train[1]) + 1
    # model_options['ydim'] = ydim
    # train[1] gives the 2nd value in the tuple: train_set_y, which is a list of Y values
    # Y values should always >= 0, since it is used for indexing
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

# layers = {'lstm': (param_init_lstm, lstm_layer)}
# the values in this dict is functions
def get_layer(name):
    fns = layers[name]
    return fns

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

"""
####################################################################################

LSTM Functions

####################################################################################
"""

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    """
    LSTM layer function
    performs calculations on the layer, per step-wise

    tparams: theano parameter set
             OrderedDict() with elements as theano.shared()
    state_below: from emb matrix
    options: hyper parameters; dict format; dict of all model hyper parameters
    prefix: 
    mask: theano tensor variable; Sequence mask; from imdb.prepare_data
          mask marked the length of each sentence on the X matrix
    
    returns
    ----
    perform one iteration through X matrix, returns a list of hidden value generated
    """
    nsteps = state_below.shape[0]
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
        x_: as W*Xt + b
        h_: hidden state vector
        c_: past state of LSTM cell
        '''
        # tparams[_p(prefix, 'U')] 's dimension is 3
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        # i, f, o, c are all concantinated together
        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))
        # c as the state of LSTM cell
        c = f * c_ + i * c
        # using mask to get rid of 'out of sentence' values
        # m_[:, None] here None adds one more dimension to mask matrix
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        # h as hidden value
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    
    ####################################################################################
    
    # W*Xt + b
    # W = numpy.random.randn(ndim, ndim) * 4 <- on 3rd axis
    # ndim = options['dim_proj']
    # state_below as emb
    # state_below 's dimension: max_length_sentence, N_samples_per_batch, Y_value_dimension*1
    # state_below's last dimension matches W's dimension
    # as this is tensor operation
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
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
    # rval is h, c; rval[0] is a list of h
    return rval[0]

'''
####################################################################################
'''
# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}

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
    
    ################################
    # Embedding and Context Window #
    ################################

    ####################################################################################
    # embedding
    # 
    # LSTM matrix_embed
    # randn = numpy.random.rand(options['n_words'], options['dim_proj'])
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    # 
    # the dimension of params['Wemb'] represents the max_degree of freedoms in the system
    # 
    # options['n_words'] should equal to x.flatten() 's dimension
    # which is post-Prep_data(), as (length_sentences, N_sentence_per_batch)
    # setting maxlen=25 and batch_size=16 gives options['n_words'] = 400
    # 
    # from later, x = [train[0][t]for t in train_index]; 
    # then before load into Model: x, mask, y = prepare_data(x, y); axis swaped
    # then emb = tparams['Wemb'][x.flatten()].reshape([ n_timesteps, n_samples, options['dim_proj'] ])
    #           as (max length of each sentence), (number of sentences of each minibatch), (word_embeding dimension)
    # 
    # into lstm_layer() as state_below
    # then n_samples = state_below.shape[1] = emb.shape[1] 
    #                                       as n_timesteps as number of sentences of each minibatch
    # then tensor.alloc(numpy_floatX(0.),n_samples,dim_proj) as _h and _c matrix
    # 
    # n_timesteps = x.shape[0] => as max length of each sentence
    # n_samples = x.shape[1] => as number of sentences of each minibatch 
    # 
    # Here LSTM assuming a context_window size of 1; 
    # since we are using LSTM, there is little point of context_window_size > 1; 
    # 
    emb = tparams['Wemb'][x.flatten()].reshape([ n_timesteps, n_samples, options['dim_proj'] ])

    ####################################################################################

    # layers = {'lstm': (param_init_lstm, lstm_layer)}
    # proj calls a LSTM 'layer' functions
    # which calculates the hidden state value and cell state value through one iteration on X matrix
    # which returns only the hidden values
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    # using mask[:,:,None] to get rid of extra 'out of sentence' hidden values
    if options['encoder'] == 'lstm':
        # sum of hidden values alone each sentences
        proj = (proj * mask[:, :, None]).sum(axis=0) # element-wise operation
        # numerical average of hidden values alone each sentences
        proj = proj / mask.sum(axis=0)[:, None]
    # if not use dropout_layer() slightly faster, but worse test error. 
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    # calculate results/predictions from outputted hidden layer values
    # softmax in the range of [0,1]
    # pred is a matrix, its axis=1 dimension is by ydim = numpy.max(train[1]) + 1
    # e.g., if y is on 0 or 1, then ydim = max(0,1) + 1 = 2
    pred = tensor.nnet.softmax( tensor.dot(proj, tparams['U']) + tparams['b'] )

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    # negtive log likelihood
    # [tensor.arange(n_samples), y], this way of indexing means 
    # y's value should always be >= 0 
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
    
    saveto='lstm_model.npz',  
    # The best model will be saved there
    loadfrom = 'lstm_model.npz',
    # load parameters from .npz files
    dataset='tweetText_tagScore.pkl',
    # name of the data set; also name of the .pkl file
    dataset_path="../Data/DataSet_Tokenize",

    # path to the data set  
    validFreq=100,  # Compute the validation error after this number of update.
    saveFreq=100,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
                          # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worse test error
                       # This frequently need a bigger model.
    reload_model=False,  
    # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    # this will get all the varialbe:value from def train_lstm into a dict
    model_options = locals().copy()
    print("Initial model options", model_options)
    # functions of loading and preparing dataset
    load_data =  LSTM_DataPrep.load_data
    prepare_data = LSTM_DataPrep.prepare_data

    print('Loading data')
    # LSTM_DataPrep.load_data(dataset, path_dataset, n_words=60000, valid_portion=0.1, maxlen=None, 
    #                         sort_by_len=True) 
    train, valid, test = load_data(dataset=dataset, path_dataset=dataset_path, 
                                   n_words=n_words, valid_portion=0.05, maxlen=maxlen)
    # adjust the size of test_set used
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random size example.  
        # So we must select a random selection of the examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
    # train[1] gives the 2nd value in the tuple: train_set_y, which is a list of Y values
    # e.g. ydim = max(0,1)+1 = 2
    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

    # saving model_options to .json file
    json_name = saveto[:-4] + '.json'
    with open(json_name, 'w') as fp:
        json.dump(model_options, fp, sort_keys=True, indent=4)

    ##################
    # Building Model #
    ##################
    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)
    # loading previous parameters or not
    if reload_model == True:
        # path = name.npz, in the save address as this code
        params = load_params(path=loadfrom, params=params)
        json_name = saveto[:-4] + '_PredReload.json'
        with open(json_name, 'w') as fp:
            json.dump(model_options, fp, sort_keys=True, indent=4)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
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
    # sgd, adadelta and rmsprop available
    # cost is a theano variable/function created at build_model(tparams, model_options)
    if optimizer == 'adadelta':
        optimizer=adadelta
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)

    ##################
    # Training Model #
    ##################
    print('Optimization')
    # kf_... is a list of the indexes of samples per each minibatch of all minibatches
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
            # Get new shuffled index for the training set.
            # kf_... is a list of the indexes of samples per each minibatch of all minibatches
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                # set use_noise for dropout_layer()
                use_noise.set_value(1.)
                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                # LSTM_DataPrep.prepare_data(seqs, labels, maxlen=None)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                # cost is a theano variable/function created at build_model(tparams, model_options)
                # here cost is calculated using the selected gradiant method
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
                    # effectively counting how many validation periods been finished
                    history_errs.append([valid_err, test_err])                      
                    
                    # update best_p
                    # valid_err or test_err or initialize
                    if (best_p is None) or (valid_err<numpy.array(history_errs)[:-1,0].min()) or (test_err<numpy.array(history_errs)[:-1,1].min()):
                        # renew best_p
                        best_p = unzip(tparams)
                        
                        # saving best_p to 'saveto'
                        print('Saving...')
                        params = best_p
                        # saving model_options (hyper-parameters) only
                        numpy.savez(saveto, history_errs=history_errs, **params)
                        # saving model_options (hyper-parameters) only
                        pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                        print('Done')
                        
                        # reset bad_counter
                        bad_counter = 0
                        print('best_p updated; bad_counter reset to 0;')
                        best_p_str = "best_p Train: %f, Valid: %f, Test: %f" % tuple( [train_err]+[valid_err]+[test_err] )
                        print(best_p_str)               
                    
                    # Early Stop
                    # eidx > 3; wait for at least 3 epochs (3 times over entire dataset) 
                    # update bad_counter when either validation or test got worse
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
    # return dict() of model_options
    print("Final model options", model_options) 
    return params


"""
####################################################################################

Prediction Codes

model options: {
    'dataset': 'trainAgainst_hillary.pkl', 
    'loadfrom': 'lstm_model_trainAgainst_hillary.npz', 
    'validFreq': 370, 
    'n_words': 12000, 
    'batch_size': 16, 
    'decay_c': 0.0, 
    'patience': 3, 
    'reload_model': None, 
    'lrate': 0.0001, 
    'max_epochs': 100, 
    'dispFreq': 10, 
    'encoder': 'lstm',

    'optimizer': <function adadelta at 0x0000000014C54C18>, 

    'valid_batch_size': 64, 
    'use_dropout': True, 
    'dim_proj': 256, 
    'maxlen': 100, 
    'saveto': 'lstm_model_trainAgainst_hillary.npz', 
    'noise_std': 0.0, 
    'test_size': -1, 
    'saveFreq': 1110, 
    'dataset_path': '../Data/DataSet_Tokenize/'
}

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
    # loading previous parameters or not
    if reload_model == True :
        # load model_options from .json file
        model_options_fileName = loadfrom[:-4] + '.json'
        with open(model_options_fileName, 'r') as fpr:
            model_options = json.load(fpr)  
        # saving reloaded model_options (hyper-parameters) to .json
        json_name = loadfrom[:-4] + '_PredReload.json'
        with open(json_name, 'w') as fpl:
            json.dump(model_options, fpl, sort_keys=True, indent=4)
        # load matrix parameters 
        params = init_params(model_options)
        params = load_params(path=loadfrom, params=params)

    # directly from training
    if params_direct is not None:
        params = params_direct
        print("params set to params_direct")

    # load model_options (dict) from train_lstm
    n_words = model_options['n_words']
    maxlen = model_options['maxlen']

    print("model parameter check")
    print("N_words: ", n_words, "dim_proj: ", model_options['dim_proj'], "maxlen: ", maxlen)
    
    # dataset path
    dataset = dataset
    dataset_path = dataset_path

    # size of each batch during prediction process
    batch_size = model_options['batch_size']

    ####################################################################################
    
    #############
    # load Data #
    #############

    # data prep functions
    load_data = LSTM_DataPrep.load_data_for_prediction
    prepare_data = LSTM_DataPrep.prepare_data

    print('Loading data')
    dataset4prediction, fileNames, scores_tag = load_data(dataset=dataset, path_dataset=dataset_path, 
                                   n_words=n_words, maxlen=maxlen)

    # dataset4prediction[1] gives the 2nd value in the tuple: train_set_y, which is a list of Y values
    # e.g. ydim = max(0,1)+1 = 2
    ydim = numpy.max(dataset4prediction[1]) + 1
    if model_options['ydim'] != ydim:
        print( "Error: prediction data set ydim NOT MATCHING training data set ydim" )
        print( "training data set ydim: %i" % model_options['ydim'] )
        print( "prediction data set ydim: %i" % ydim )
        # force exit
        return None

    ##################
    # Building Model #
    ##################
    print('Building model')

    # This create Theano Shared Variable from the parameters.
    tparams = init_tparams(params)
    
    # build_model()
    (use_noise, x, mask, y, 
        f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    ######################
    # Predicting Results #
    ######################
    print('Making Predictions')
    print("%d prediction examples" % len( dataset4prediction[0] ) )

    # all the data, np.array format
    dataset_sentences_digit = []
    dataset_preds_prob = []
    dataset_preds = []

    try:
        # pred_index_batches_... is a list of the indexes (list) 
        # of samples per each minibatch of all minibatches
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
            # Get the data in numpy.ndarray format
            # This swap the axis!
            # Return something of shape (minibatch maxlen, n samples)
            # LSTM_DataPrep.prepare_data(seqs, labels, maxlen=None)
            x, mask, y = prepare_data(x, y)
            n_samples += x.shape[1]
            # n_samples cumulated through all minibatches
            print('Seen %d samples' % n_samples)            
            
            # pred = tensor.nnet.softmax( tensor.dot(proj, tparams['U']) + tparams['b'] )
            # f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
            # f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
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

    # MySQL_DBkey = {'host':'localhost', 'user':'sa', 'password':'fanyu01', 'db':'ultrajuly_p1v5_p2v2','charset':'utf8mb4'}
    MySQL_DBkey = {'host':'localhost', 'user':'sa', 'password':'fanyu01', 'db':'ultrajuly_p1v5_p2v2','charset':'utf8'}

    ####################################################################################
    dict_tokenizeParameters_trainAgainst_trump = {
        'dataset':'trainAgainst_trump', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Training/',
        'tokenizer_path': './scripts/tokenizer/',
        # same for all cases
        'lstm_saveto': 'lstm_TA_trump',
        'lstm_loadfrom':'lstm_TA_trump',
        # LSTM model parameter save/load
        'Yvalue_list':['posi_trump', 'neg_trump'],
        # root name for cases to be considered
        'posi_trump_folder':['posi_neut', 'posi_neg'],
        'neg_trump_folder':['neg_posi', 'neg_neut', 'neg_neg'],
        
        'posi_trump_score':1,
        'neg_trump_score':0
        }
    
    dict_tokenizeParameters_trainAgainst_hillary = {
        'dataset':'trainAgainst_hillary', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Training/',
        'tokenizer_path': './scripts/tokenizer/',
        # same for all cases
        'lstm_saveto': 'lstm_TA_hillary',
        'lstm_loadfrom':'lstm_TA_hillary',
        # LSTM model parameter save/load
        'Yvalue_list':['posi_hillary', 'neg_hillary'],
        # root name for cases to be considered
        'posi_hillary_folder':['neut_posi', 'neg_posi'],
        'neg_hillary_folder':['posi_neg', 'neut_neg', 'neg_neg'],
        
        'posi_hillary_score':1,
        'neg_hillary_score':0
        }

    dict_tokenizeParameters_trainAgainst_trumphillary = {
        'dataset':'trainAgainst_trumphillary', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Training/',
        'tokenizer_path': './scripts/tokenizer/',
        # same for all cases
        'lstm_saveto': 'lstm_TA_trumphillary',
        'lstm_loadfrom':'lstm_TA_trumphillary',
        # LSTM model parameter save/load
        'Yvalue_list':['trump', 'hillary', 'neutral'],
        # root name for cases to be considered
        'trump_folder':['posi_neut', 'posi_neg'],
        'hillary_folder':['neut_posi', 'neg_posi'],
        'neutral_folder':['neg_neg', 'neut_neg', 'neg_neut', 'neut_neut'],
        
        'trump_score':2,
        'hillary_score':0,
        'neutral_score':1
        }
    
    ####################################################################
    # training
    flag_training = True 
    dict_training = dict_tokenizeParameters_trainAgainst_trumphillary

    ####################################################################
    # predicting
    flag_predicting = True 
    
    pred_columnName = 'TA_trumphillary_P1N0'
    sql_tableName = 'filtered_tweet_stack_Nword5000_Dim1024'

    path_preToken_Predicting = '../Data/DataSet_Predicting/'

    dict_tokenizeParameters_trainAgainst_ = {
        'dataset':'trainAgainst_trumphillary', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Predicting/',
        'tokenizer_path': './scripts/tokenizer/',
        # same for all cases
        'lstm_saveto': 'lstm_TA_trumphillary_Nwords_5000_dimProj_1024',
        'lstm_loadfrom':'lstm_TA_trumphillary_Nwords_5000_dimProj_1024',
        # LSTM model parameter save/load
        'Yvalue_list':['trump', 'hillary', 'neutral'],
        # root name for cases to be considered
        'trump_folder':['posi_neut', 'posi_neg'],
        'hillary_folder':['neut_posi', 'neg_posi'],
        'neutral_folder':['neg_neg', 'neut_neg', 'neg_neut', 'neut_neut'],
        
        'trump_score':2,
        'hillary_score':0,
        'neutral_score':1
        }
    dict_predicting = dict_tokenizeParameters_trainAgainst_
    
    ####################################################################
    # for training !!!
    if flag_training == True:

        para_dataset = dict_training['dataset'] + '.pkl'
        para_dataset_path = dict_training['dataset_path']

        para_n_words = 5000
        dim_proj = 1024

        para_saveto = dict_training['lstm_saveto']
        para_saveto = para_saveto + '_Nwords_' + str(para_n_words)
        para_saveto = para_saveto + '_dimProj_' + str(dim_proj)
        para_saveto = para_saveto + '.npz'

        para_loadfrom = para_saveto

        params = train_lstm(
            max_epochs=100,
            patience=10, # Number of validation period to wait before early stop if no progress
            saveFreq=200,
            validFreq=200,
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
        dict_predicting['dataset_path'] = path_preToken_Predicting

        # setting correct 9 folders to the class with highest Y-value
        full_folder_list = ['posi_posi', 'posi_neut', 'posi_neg',
                            'neut_posi', 'neut_neut', 'neut_neg',
                            'neg_posi', 'neg_neut', 'neg_neg']
        # thus passing the Y-value_max into LSTM
        # and setting all other folders to [], avoiding overlapping data

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





