
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

def Tokenize_Main(path_preToken_DataSet, path_tokenizer):

	# get current address
	currdir = os.getcwd()
	print 'currdir: ', currdir

	path_preToken_DataSet = os.path.join(currdir, path_preToken_DataSet)
	print "path_preToken_DataSet: \n", path_preToken_DataSet

	path_tokenizer = os.path.join(currdir, path_tokenizer)
	print "path_tokenizer: \n", path_tokenizer

	DataSet_Pickle_Main(DataSet_preToken_Path=path_preToken_DataSet,
						path_tokenizer=path_tokenizer)

"""
########################################################################################
"""

if __name__ == '__main__':
	
	path_preToken_DataSet = '../Data/DataSet_Tokenize/'
	path_tokenizer = './scripts/tokenizer/'

	Tokenize_Main(path_preToken_DataSet=path_preToken_DataSet, path_tokenizer=path_tokenizer)

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








