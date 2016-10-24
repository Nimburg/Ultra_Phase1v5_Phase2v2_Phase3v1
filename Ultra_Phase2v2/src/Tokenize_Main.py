
import numpy
from collections import OrderedDict
import os
import glob

import cPickle as pkl
from subprocess import Popen, PIPE

import sys
currdir = os.getcwd()
path = currdir + '/scripts/tokenizer'
print "path to DataSet_Pickle_Main: ", path
sys.path.insert(0, path)
from DataSet_Pickle import DataSet_Pickle_Main

'''
####################################################################
'''

def Tokenize_Main(dict_parameters):

	# get current address
	currdir = os.getcwd()
	print 'currdir: ', currdir

	# path_preToken_DataSet = '../Data/DataSet_Tokenize/'
	path_preToken_DataSet = dict_parameters['dataset_path']
	path_preToken_DataSet = os.path.join(currdir, path_preToken_DataSet)
	print "path_preToken_DataSet: \n", path_preToken_DataSet

	# path_tokenizer = './scripts/tokenizer/'
	path_tokenizer = dict_parameters['tokenizer_path']
	path_tokenizer = os.path.join(currdir, path_tokenizer)
	print "path_tokenizer: \n", path_tokenizer

	# Data Pickle Operation
	N_uniqueWords = DataSet_Pickle_Main(dict_parameters=dict_parameters,
						DataSet_preToken_Path=path_preToken_DataSet,
						path_tokenizer=path_tokenizer)

	# return to currdir
	os.chdir(currdir)

	return N_uniqueWords

"""
########################################################################################
"""

if __name__ == '__main__':

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

	Tokenize_Main(dict_parameters = dict_tokenizeParameters_trainAgainst_trump)








'''
Number of sentences before tokenize: 2121
Tokenizing..
Tokenizer Version 1.1
Language: en
Number of threads: 1
Done
Number of sentences after tokenize: 2121
Building dictionary..
31114  total words  5218  unique words



number of cases of score 1: 1037
number of cases of score -1: 223
number of cases of score 0: 861

number of cases of score 1: 255
number of cases of score -1: 56
number of cases of score 0: 182


'''








