
import numpy
from collections import OrderedDict
import os
import glob

import cPickle as pkl
from subprocess import Popen, PIPE


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

def build_dict(path_data, path_tokenizer, path_folderList):
	'''
	path: the universal 'path' variable + train or test

	Note: to build the dictionary, it is actually OK to use all the cases
	'''
	print 'build_dict(data set path): ', path_data
	sentences = []
	
	####################################################################
	# go through all necessary folders
	# path_folderList = ['%s/posi_neut/','%s/posi_neg/','%s/neut_posi/', etc]
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

def DataSet_Pickle_Main(dict_parameters, DataSet_preToken_Path, path_tokenizer):
	'''
	DataSet_preToken_Path: '../Data/DataSet_Tokenize/'
	path_tokenizer: './scripts/tokenizer/'

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
	'''
	###############################################################################
	# get all related folders' list for build_dict()
	path_folderList_dict = []
	for cases in dict_parameters['Yvalue_list']:
		name_caseFolder_list = cases + '_folder'
		path_folderList_dict = path_folderList_dict + dict_parameters[name_caseFolder_list]
	print 'List of Folders involved for dictionary building: ', path_folderList_dict
	
	# convert into correct format: '%s/posi_neut/'
	for idx in range(len(path_folderList_dict)):
		new_folder = '%s/'+path_folderList_dict[idx]+'/'
		path_folderList_dict[idx] = new_folder

	# build dictionary
	dictionary = build_dict(path_data=os.path.join(DataSet_preToken_Path, 'train'), 
							path_tokenizer=path_tokenizer, path_folderList=path_folderList_dict)

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
	# go through dict_parameters['Yvalue_list']
	for case in dict_parameters['Yvalue_list']:
		# dict names
		name_caseFolder_list = case + '_folder'
		name_case_score = case + '_score'
		# folders and score for case
		case_folderList = dict_parameters[name_caseFolder_list]
		case_score = dict_parameters[name_case_score]
		# for this case, grab data
		train_x, train_y, Ncase =folderList_grab_data(path_folderList=case_folderList, 
													  trainOrtest='train/', 
													  vari_X=train_x, vari_Y=train_y, 
													  case_score=case_score)
		print "number of cases of %s of score %i: %i" % tuple( [case] + [case_score] + [Ncase] )
		print "size of training set X&Y: %i, %i" % tuple( [len(train_x)] + [len(train_y)] )

	###############################################################################
	# test data set
	test_x = []
	test_y = []
	# go through dict_parameters['Yvalue_list']
	for case in dict_parameters['Yvalue_list']:
		# dict names
		name_caseFolder_list = case + '_folder'
		name_case_score = case + '_score'
		# folders and score for case
		case_folderList = dict_parameters[name_caseFolder_list]
		case_score = dict_parameters[name_case_score]
		# for this case, grab data
		test_x, test_y, Ncase =folderList_grab_data(path_folderList=case_folderList, 
													  trainOrtest='test/', 
													  vari_X=test_x, vari_Y=test_y, 
													  case_score=case_score)
		print "number of cases of %s of score %i: %i" % tuple( [case] + [case_score] + [Ncase] )
		print "size of testing set X&Y: %i, %i" % tuple( [len(test_x)] + [len(test_y)] )
	
	###############################################################################
	# outputs
	# X and Y, train and test data sets
	# 2 * pkl.dump()
	dict_parameters['dataset']
	imdb_pkl_fileName = DataSet_preToken_Path + dict_parameters['dataset'] + '.pkl'
	f = open(imdb_pkl_fileName, 'wb')
	pkl.dump((train_x, train_y), f, -1)
	pkl.dump((test_x, test_y), f, -1)
	f.close()

	# dictionary
	imdb_dict_pkl_fileName = DataSet_preToken_Path + dict_parameters['dataset'] + '.dict.pkl'
	f = open(imdb_dict_pkl_fileName, 'wb')
	pkl.dump(dictionary, f, -1)
	f.close()

	# return values
	# number of unique words
	return len(dictionary)

"""
########################################################################################
"""










