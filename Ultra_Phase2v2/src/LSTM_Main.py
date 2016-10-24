
from __future__ import print_function
import six.moves.cPickle as pickle


from collections import OrderedDict
import sys
import time
import numpy

import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# data preparation file
import LSTM_DataPrep


"""
####################################################################################

Public Variables and Functions

####################################################################################
"""

# datasets = {'tweetText_tupleScores': (LSTM_DataPrep.load_data, LSTM_DataPrep.prepare_data)}

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

# call functions from LSTM_DataPrep
# returns 2 functions
def get_dataset(name):
	'''
	returns 2 functions
	datasets = {'tweetText_tupleScores': (LSTM_DataPrep.load_data, LSTM_DataPrep.prepare_data)}
	'''
	# return datasets[name][0], datasets[name][1]
	return LSTM_DataPrep.load_data, LSTM_DataPrep.prepare_data

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
	# embedding
	randn = numpy.random.rand(options['n_words'], 
							  options['dim_proj'])
	params['Wemb'] = (0.01 * randn).astype(config.floatX)
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
	# slice the concatenated W matrix
	def _slice(_x, n, dim):
		'''
		minibatch creator
		n: index of variable matrix
		dim: size of variable matrix
		'''
		if _x.ndim == 3:
			return _x[:, :, n * dim:(n + 1) * dim]
		return _x[:, n * dim:(n + 1) * dim]
	# one step of the LSTM operation
	def _step(m_, x_, h_, c_):
		'''
		m_: mask
		x_: as W*Xt + b
		h_: hidden state vector
		c_: past state of LSTM cell
		'''
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
	# W*Xt + b
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
	# 
	emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
												n_samples,
												options['dim_proj']])
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
	
	patience=3,  # Number of epoch to wait before early stop if no progress
	max_epochs=5000,  # The maximum number of epoch to run
	dispFreq=10,  # Display to stdout the training progress every N updates
	decay_c=0.,  # Weight decay for the classifier applied to the U weights.
	lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
	
	n_words=1000,  
	# Vocabulary size

	optimizer=adadelta,  
	# sgd, adadelta and rmsprop available, 
	# sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
	# optimizer's value is a function

	encoder='lstm',  # TODO: can be removed must be lstm.
	
	saveto='lstm_model.npz',  
	# The best model will be saved there
	loadfrom = 'lstm_model.npz',
	# load parameters from .npz files
	
	validFreq=370,  # Compute the validation error after this number of update.
	saveFreq=1110,  # Save the parameters after every saveFreq updates
	maxlen=100,  # Sequence longer then this get ignored
	batch_size=16,  # The batch size during training.
	valid_batch_size=64,  # The batch size used for validation/test set.
	
	dataset='tweetText_tagScore.pkl',
	# name of the data set; also name of the .pkl file
	# datasets = {'Name?': (LSTM_DataPrep.load_data, LSTM_DataPrep.prepare_data)}
	dataset_path="../Data/DataSet_Tokenize",
	# path to the data set

	# Parameter for extra option
	noise_std=0.,
	use_dropout=True,  # if False slightly faster, but worse test error
					   # This frequently need a bigger model.
	reload_model=None,  
	# Path to a saved model we want to start from.
	test_size=-1,  # If >0, we keep only this number of test example.
):

	# Model options
	# this will get all the varialbe:value from def train_lstm into a dict
	model_options = locals().copy()
	print("Initial model options", model_options)
	# get data from dataset named as 'dataset'
	# datasets = {'tweetText_tupleScores': (LSTM_DataPrep.load_data, LSTM_DataPrep.prepare_data)}
	load_data, prepare_data = get_dataset(dataset)
	print('Loading data')
	# LSTM_DataPrep.load_data(dataset, path_dataset, n_words=60000, valid_portion=0.1, maxlen=None, 
	# 						  sort_by_len=True)	
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

	##################
	# Building Model #
	##################
	print('Building model')
	# This create the initial parameters as numpy ndarrays.
	# Dict name (string) -> numpy ndarray
	params = init_params(model_options)
	# loading previous parameters or not
	if reload_model:
		# path = name.npz, in the save address as this code
		load_params(path=loadfrom, params=params)
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
				# save parameters
				if saveto and numpy.mod(uidx, saveFreq) == 0:
					print('Saving...')
					if best_p is not None:
						params = best_p
					else:
						params = unzip(tparams)
					numpy.savez(saveto, history_errs=history_errs, **params)
					pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
					print('Done')

				# validation
				if numpy.mod(uidx, validFreq) == 0:
					use_noise.set_value(0.)
					train_err = pred_error(f_pred, prepare_data, train, kf)
					valid_err = pred_error(f_pred, prepare_data, valid,
										   kf_valid)
					test_err = pred_error(f_pred, prepare_data, test, kf_test)

					history_errs.append([valid_err, test_err])

					if (best_p is None or
						valid_err <= numpy.array(history_errs)[:,
															   0].min()):

						best_p = unzip(tparams)
						bad_counter = 0

					print( ('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err) )

					if (len(history_errs) > patience and
						valid_err >= numpy.array(history_errs)[:-patience,
															   0].min()):
						bad_counter += 1
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

	print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
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
	return model_options

"""
####################################################################################

Load Parameter and Prediction Codes

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

def load_predict_lstm(model_options, dataset, dataset_path):

	####################################################################################
	# load model_options (dict) from train_lstm
	n_words = model_options['n_words']	
	maxlen = model_options['maxlen']
	# parameter dict path
	loadfrom = model_options['loadfrom']
	# dataset path
	dataset = dataset
	dataset_path = dataset_path

	reload_model = True # naturally
	# size of each batch during prediction process
	batch_size = model_options['batch_size']

	####################################################################################
	# data prep functions
	load_data = LSTM_DataPrep.load_data_for_prediction
	prepare_data = LSTM_DataPrep.prepare_data

	print('Loading data')
	dataset4prediction = load_data(dataset=dataset, path_dataset=dataset_path, 
								   n_words=n_words, maxlen=maxlen)

	# dataset4prediction[1] gives the 2nd value in the tuple: train_set_y, which is a list of Y values
	# e.g. ydim = max(0,1)+1 = 2
	ydim = numpy.max(dataset4prediction[1]) + 1
	if model_options['ydim'] != ydim:
		print "Error: prediction data set ydim NOT MATCHING training data set ydim"
		print "training data set ydim: %i" % model_options['ydim']
		print "prediction data set ydim: %i" % ydim
		# force exit
		return None

	##################
	# Building Model #
	##################
	print('Building model')
	# This create the initial parameters as numpy ndarrays.
	params = init_params(model_options)
	# loading previous parameters or not
	if reload_model:
		load_params(path=loadfrom, params=params)
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
			# f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
			# f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
			preds_prob = f_pred_prob(x, mask)
			preds = f_pred(x, mask)
			# end of this minibatch, n_samples cumulated through all minibatches
			print('Seen %d samples' % n_samples)





	except KeyboardInterrupt:
		print("Training interupted")







"""
####################################################################################

Test Codes

####################################################################################
"""

if __name__ == '__main__':
	'''
	# See function train for all possible parameter and there definition.
	train_lstm(
		max_epochs=100,
		test_size=-1, 
		# If >0, we keep only this number of test example.
		
		dim_proj=256, # word embeding dimension and LSTM number of hidden units.
		n_words=1000, # Vocabulary size
		
		dataset='tweetText_tagScore.pkl',
		# name of the data set, 'sth.pkl'; also name of the .pkl file
		# datasets = {'Name?': (LSTM_DataPrep.load_data, LSTM_DataPrep.prepare_data)}
		dataset_path="../Data/DataSet_Tokenize",
		# path to the data set

		saveto='lstm_model.npz',
		loadfrom = 'lstm_model.npz',
		reload_model=None # whether reload revious parameter or not

	)
	'''
	####################################################################################
	dict_tokenizeParameters_trainAgainst_trump = {
		'dataset':'trainAgainst_trump', 
		# PLUS .pkl or dict.pkl for LSTM
		'dataset_path': '../Data/DataSet_Tokenize/',
		'tokenizer_path': './scripts/tokenizer/',
		# same for all cases
		'lstm_saveto': 'lstm_model_trainAgainst_trump.npz',
		'lstm_loadfrom':'lstm_model_trainAgainst_trump.npz',
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
		'dataset_path': '../Data/DataSet_Tokenize/',
		'tokenizer_path': './scripts/tokenizer/',
		# same for all cases
		'lstm_saveto': 'lstm_model_trainAgainst_hillary.npz',
		'lstm_loadfrom':'lstm_model_trainAgainst_hillary.npz',
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
		'dataset_path': '../Data/DataSet_Tokenize/',
		'tokenizer_path': './scripts/tokenizer/',
		# same for all cases
		'lstm_saveto': 'lstm_model_trainAgainst_trumphillary.npz',
		'lstm_loadfrom':'lstm_model_trainAgainst_trumphillary.npz',
		# LSTM model parameter save/load
		'Yvalue_list':['trump', 'hillary', 'neutral'],
		# root name for cases to be considered
		'trump_folder':['posi_neut', 'posi_neg'],
		'hillary_folder':['neut_posi', 'neg_posi'],
		'neutral_folder':['neg_neg'],
		
		'trump_score':2,
		'hillary_score':0,
		'neutral_score':1,
		}

	####################################################################################
	para_dataset = dict_tokenizeParameters_trainAgainst_hillary['dataset'] + '.pkl'
	para_dataset_path = dict_tokenizeParameters_trainAgainst_hillary['dataset_path']

	para_n_words = 12000
	# 230751  total words  19502  unique words
	# 177728  total words  15751  unique words
	# 43113  total words  6723  unique words

	para_saveto = dict_tokenizeParameters_trainAgainst_hillary['lstm_saveto']
	para_loadfrom = para_saveto

	model_options = train_lstm(
		max_epochs=100,
		test_size=-1, 
		# If >0, we keep only this number of test example.
		
		dim_proj=256, # word embeding dimension and LSTM number of hidden units.
		n_words=para_n_words, # Vocabulary size
		
		dataset=para_dataset,
		# name of the data set, 'sth.pkl'; also name of the .pkl file
		# datasets = {'Name?': (LSTM_DataPrep.load_data, LSTM_DataPrep.prepare_data)}
		dataset_path=para_dataset_path,
		# path to the data set

		saveto=para_saveto,
		loadfrom = para_loadfrom,
		reload_model=None # whether reload revious parameter or not
	)












'''
####################################################################################

dict_tokenizeParameters_trainAgainst_trump
dim_proj=256
n_words=10000 out of 19502

Epoch  0 Update  370 Cost  0.359852433205
('Train ', 0.098391401572531234, 'Valid ', 0.0931506849315068, 'Test ', 0.096908939014202167)
Epoch  0 Update  380 Cost  0.404751211405

Epoch  4 Update  4070 Cost  0.0156991314143
('Train ', 0.0098824208324316265, 'Valid ', 0.024657534246575352, 'Test ', 0.028682818156502421)
Epoch  4 Update  4080 Cost  0.0406259484589

Epoch  5 Update  5180 Cost  0.0187643319368
('Train ', 0.0038231263074370858, 'Valid ', 0.020547945205479423, 'Test ', 0.021442495126705707)
Epoch  5 Update  5190 Cost  0.0154427438974

####################################################################################

dict_tokenizeParameters_trainAgainst_hillary
dim_proj=256
n_words=10000 out of 15751

number of cases of posi_hillary of score 1: 345
size of training set X&Y: 345, 345
number of cases of neg_hillary of score 0: 11393
size of training set X&Y: 11738, 11738

Epoch  0 Update  370 Cost  0.232056573033
('Train ', 0.029544633176725732, 'Valid ', 0.024013722126929649, 'Test ', 0.027583914921900932)
Epoch  0 Update  380 Cost  0.021654073149

Epoch  3 Update  2220 Cost  0.0270751230419
('Train ', 0.02041922659920492, 'Valid ', 0.015437392795883409, 'Test ', 0.019607843137254943)
Epoch  3 Update  2230 Cost  0.0829429253936

Epoch  6 Update  4810 Cost  0.0127623127773
('Train ', 0.0035236718467654971, 'Valid ', 0.018867924528301883, 'Test ', 0.015952143569292088)
Epoch  6 Update  4820 Cost  0.00353421084583

Epoch  11 Update  8140 Cost  0.00812957342714
('Train ', 0.00054210336104087986, 'Valid ', 0.015437392795883409, 'Test ', 0.011631771352608844)
Early Stop!

####################################################################################

dict_tokenizeParameters_trainAgainst_trumphillary
dim_proj=256
n_words=4500 out of 6723

number of cases of trump of score 2: 1432
size of training set X&Y: 1432, 1432
number of cases of hillary of score 0: 341
size of training set X&Y: 1773, 1773
number of cases of neutral of score 1: 1187
size of training set X&Y: 2960, 2960

Epoch  2 Update  370 Cost  1.01258981228
('Train ', 0.4939544807965861, 'Valid ', 0.46621621621621623, 'Test ', 0.48753462603878117)
Epoch  2 Update  380 Cost  0.82085442543

Epoch  12 Update  2220 Cost  0.0179200656712
('Train ', 0.0024893314366998265, 'Valid ', 0.22297297297297303, 'Test ', 0.20360110803324105)
Epoch  12 Update  2230 Cost  0.0266376640648

Epoch  23 Update  4070 Cost  0.00119850691408
('Train ', 0.0, 'Valid ', 0.2567567567567568, 'Test ', 0.23684210526315785)
Early Stop!

####################################################################################

dict_tokenizeParameters_trainAgainst_trumphillary
dim_proj=256
n_words=6000 out of 6723

number of cases of trump of score 2: 1432
size of training set X&Y: 1432, 1432
number of cases of hillary of score 0: 341
size of training set X&Y: 1773, 1773
number of cases of neutral of score 1: 1187
size of training set X&Y: 2960, 2960

Epoch  2 Update  370 Cost  1.11635255814
('Train ', 0.51066856330014221, 'Valid ', 0.4932432432432432, 'Test ', 0.50969529085872578)
Epoch  2 Update  380 Cost  0.882966160774

Epoch  8 Update  1480 Cost  0.196475118399
('Train ', 0.012446657183499243, 'Valid ', 0.19594594594594594, 'Test ', 0.19390581717451527)
Epoch  8 Update  1490 Cost  0.0531963817775

Epoch  21 Update  3700 Cost  0.000893165648449
('Train ', 0.00035561877667145136, 'Valid ', 0.26351351351351349, 'Test ', 0.25484764542936289)
Early Stop!








'''








