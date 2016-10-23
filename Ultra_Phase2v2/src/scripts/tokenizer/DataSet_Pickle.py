
import numpy
from collections import OrderedDict
import os
import glob

import cPickle as pkl
from subprocess import Popen, PIPE

'''
####################################################################

The LSTM could not easily handle a complex variable situation 
(with real and imaginery parts representing scores for keyword 1 and 2)

It is not practical either to add one more dimension directly, 
making W, U etc into theano tensor3 while b from vector to matrix

Nor is it very helpful to 'flatten' the extra dimension
putting scores (int, int) into extra columns or rows


Thus, the solution adopted here is to:

use +2 representing support for keyword 1 'trump'
(+1, 0 or -1) #2

use -2 representing support for keyword 2 'hillary'
(0 or -1, +1) #2

use 0 for neg_neg cases
(-1, -1) #1

drop all other cases

####################################################################
'''

'''
####################################################################
'''

# dataset_path='./DataSet_Raw/'
tokenizer_cmd = ['tokenizer.perl', '-l', 'en', '-q', '-']

'''
####################################################################
'''

def tokenize(sentences):

	#tokenizer_cmd = ['tokenizer.perl', '-l', 'en', '-q', '-']

	print 'Tokenizing..'
	text = "\n".join(sentences)
	tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE, shell=True) # shell=True works in windows
	tok_text, _ = tokenizer.communicate(text)
	toks = tok_text.split('\n')[:-1]
	print 'Done'

	return toks

"""
########################################################################################
"""

def build_dict(path_data, path_tokenizer):
	'''
	path: the universal 'path' variable + train or test

	Note: to build the dictionary, it is actually OK to use all the cases
	'''
	print 'build_dict(data set path): ', path_data
	sentences = []
	
	####################################################################
	# for all the nine folders	
	path_folderList = ['%s/posi_neut/','%s/posi_neg/','%s/neut_posi/','%s/neg_posi/', '%s/neg_neg/']
	
	for path in path_folderList:
		# change address to posi texts
		os.chdir( path % path_data)
		for ff in glob.glob("*.txt"):
			with open(ff, 'r') as f:
				sentences.append(f.readline().strip())	

	# since tokenizer.perl is in this address
	os.chdir(path_tokenizer)
	print "Number of sentences before tokenize: %i" % len(sentences)
	sentences = tokenize(sentences)
	print "Number of sentences after tokenize: %i" % len(sentences)

	####################################################################
	# post tokenization
	print 'Building dictionary..'

	# {word token : N_calls}
	wordcount = dict() 
	for ss in sentences:
		words = ss.strip().lower().split()
		for w in words:
			if w not in wordcount:
				wordcount[w] = 1
			else:
				wordcount[w] += 1
	counts = wordcount.values() # list of N_calls of wordcount
	keys = wordcount.keys() # list of keys, or 'word token', of wordcount
	
	# {word token: ranking in terms of N_call, in decent order}
	worddict = dict()
	sorted_idx = numpy.argsort(counts)[::-1] # list of indexes of the list 'counts'
	for idx, ss in enumerate(sorted_idx):
		# keys[ss] is word token, idx+2 is its ranking in sorted_idx + 2
		worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

	print numpy.sum(counts), ' total words ', len(keys), ' unique words'

	return worddict

"""
########################################################################################
"""

def grab_data(path_data, path_tokenizer, dictionary):
	'''
	path: the universal 'path' variable + train or test + pos or neg
	dictionary: 
	'''
	sentences = []
	# change to given data set address
	os.chdir(path_data)
	for ff in glob.glob("*.txt"):
		with open(ff, 'r') as f:
			sentences.append(f.readline().strip())
	# change to tokenizer address
	os.chdir(path_tokenizer)
	sentences = tokenize(sentences)

	seqs = [None] * len(sentences)
	for idx, ss in enumerate(sentences):
		words = ss.strip().lower().split() # list of word tokens per sentence from sentences
		seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

	return seqs

"""
########################################################################################
"""

def DataSet_Pickle_Main(DataSet_preToken_Path, path_tokenizer):
	'''
	after tokenize, the scores by keyword1 and keyword2 is set into complex numbers
	real part for keyword1, imaginary part for keyword2
	'''

	# build dictionary
	dictionary = build_dict(path_data=os.path.join(DataSet_preToken_Path, 'train'), 
							path_tokenizer=path_tokenizer)

	# path to cases of 'supporting keyword1', with score +2
	path_folderList_SP2 = ['posi_neut','posi_neg']

	# path to cases of 'supporting keyword2', with score -2
	path_folderList_SN2 = ['neut_posi','neg_posi']

	# path to cases (only 1 path) of 'neutral', with score (-1, -1)
	path_folderList_neut = ['neg_neg']

	###############################################################################
	# local function used to grab_data() from given list of folders
	def folderList_grab_data(path_folderList, trainOrtest, vari_X, vari_Y, case_score):
		'''
		path_folderList: list of paths 
		trainOrtest: 'train/' or 'test/'
		vari_X: [], mother function variable
		vari_Y: []
		case_score: given, int
		'''
		counter = 0
		# go through path_folderList
		for path_folder in path_folderList:
			path_folder = trainOrtest + path_folder
			vari_X_values = grab_data(path_data=DataSet_preToken_Path+path_folder, 
									  path_tokenizer=path_tokenizer,
									  dictionary=dictionary)	
			vari_X = vari_X + vari_X_values
			vari_Y = vari_Y + [case_score] * len(vari_X_values)
			counter += len(vari_X_values)

		return vari_X, vari_Y, counter
	###############################################################################

	# training data set
	train_x = []
	train_y = []	
	
	# for 'supporting keyword1', with score +2
	train_x, train_y, Ncase =folderList_grab_data(path_folderList=path_folderList_SP2, 
												  trainOrtest='train/', 
												  vari_X=train_x, vari_Y=train_y, 
												  case_score=1)
	print "number of cases of score %i: %i" % tuple( [1] + [Ncase] )
	print "size of training set X&Y: %i, %i" % tuple( [len(train_x)] + [len(train_y)] )

	# for 'supporting keyword2', with score -2
	train_x, train_y, Ncase =folderList_grab_data(path_folderList=path_folderList_SN2, 
												  trainOrtest='train/', 
												  vari_X=train_x, vari_Y=train_y, 
												  case_score=-1)
	print "number of cases of score %i: %i" % tuple( [-1] + [Ncase] )
	print "size of training set X&Y: %i, %i" % tuple( [len(train_x)] + [len(train_y)] )

	# for 'mutral dislike', with score 0
	train_x, train_y, Ncase =folderList_grab_data(path_folderList=path_folderList_neut, 
												  trainOrtest='train/', 
												  vari_X=train_x, vari_Y=train_y, 
												  case_score=0)
	print "number of cases of score %i: %i" % tuple( [0] + [Ncase] )	
	print "size of training set X&Y: %i, %i" % tuple( [len(train_x)] + [len(train_y)] )

	################################################################
	# test data set
	test_x = []
	test_y = []

	# for 'supporting keyword1', with score +2
	test_x, test_y, Ncase =folderList_grab_data(path_folderList=path_folderList_SP2, 
												  trainOrtest='test/', 
												  vari_X=test_x, vari_Y=test_y, 
												  case_score=1)
	print "number of cases of score %i: %i" % tuple( [1] + [Ncase] )
	print "size of testing set X&Y: %i, %i" % tuple( [len(test_x)] + [len(test_y)] )

	# for 'supporting keyword2', with score -2
	test_x, test_y, Ncase =folderList_grab_data(path_folderList=path_folderList_SN2, 
												  trainOrtest='test/', 
												  vari_X=test_x, vari_Y=test_y, 
												  case_score=-1)
	print "number of cases of score %i: %i" % tuple( [-1] + [Ncase] )
	print "size of testing set X&Y: %i, %i" % tuple( [len(test_x)] + [len(test_y)] )

	# for 'mutral dislike', with score 0
	test_x, test_y, Ncase =folderList_grab_data(path_folderList=path_folderList_neut, 
												  trainOrtest='test/', 
												  vari_X=test_x, vari_Y=test_y, 
												  case_score=0)
	print "number of cases of score %i: %i" % tuple( [0] + [Ncase] )
	print "size of testing set X&Y: %i, %i" % tuple( [len(test_x)] + [len(test_y)] )
	
	################################################################
	# outputs
	# X and Y, train and test data sets
	# 2 * pkl.dump()
	imdb_pkl_fileName = DataSet_preToken_Path + 'tweetText_tagScore.pkl'
	f = open(imdb_pkl_fileName, 'wb')
	pkl.dump((train_x, train_y), f, -1)
	pkl.dump((test_x, test_y), f, -1)
	f.close()

	# dictionary
	imdb_dict_pkl_fileName = DataSet_preToken_Path + 'tweetText_tagScore.dict.pkl'
	f = open(imdb_dict_pkl_fileName, 'wb')
	pkl.dump(dictionary, f, -1)
	f.close()


"""
########################################################################################
"""










