
####################################################################
# 
# 3. import the marked tags 
# and use the marked tags to export tweet text as training/testing data sets
# separating into groups of: key1 is trump, key2 is hillary
# 
# posi_posi, posi_neut, posi_neg
# neut_posi, neut_neut, neut_neg
# neg_posi, neg_neut, neg_neg
# 
# thus each tweet text has its senti-value of (x,y) with x&y -1, 0 or 1
# 
####################################################################
'''
The LSTM could not easily handle a complex variable situation 
(with real and imaginery parts representing scores for keyword 1 and 2)

It is not practical either to add one more dimension directly, 
making W, U etc into theano tensor3 while b from vector to matrix

Nor is it very helpful to 'flatten' the extra dimension
putting scores (int, int) into extra columns or rows


Thus, the solution adopted here is to:

use +1 representing support for keyword 1 'trump'
(+1, 0 or -1) #2

use -1 representing support for keyword 2 'hillary'
(0 or -1, +1) #2 

use 0 for neg_neg cases
(-1, -1) #1

drop the (0, 0) cases, as too much noise there
'''
####################################################################
'''
Modified Designs

in TextExtract_byTags(), the following filters are applied on tweetText:

0. extract TagList_text and split by ',', putting into a list, then into a set
1. split tweetText by ' ', getting each words from a tweetText
2. eliminate strings starting with '@', those are mentioned users
3. use the set(list[tags]), to check how many non-tag words is in the tweetText
   which: not marketScore_tags, not word with keywords inside, not https:
   important to lower() tweetText
4. use a threshold to select tweets with more than a certain number of non-tag words
'''
####################################################################


import json
import os
import numpy as np 
import pandas as pd
import collections as col

import pymysql.cursors

from MarkedTag_Import import MarkedTag_Import, TextExtract_byTags
from Tokenize_Main import Tokenize_Main


"""
####################################################################
"""

# variable type check
def check_args(*types):
	def real_decorator(func):
		def wrapper(*args, **kwargs):
			for val, typ in zip(args, types):
				assert isinstance(val, typ), "Value {} is not of expected type {}".format(val, typ)
			return func(*args, **kwargs)
		return wrapper
	return real_decorator

# check each line from readline(), check length, start/end character
@check_args(str, int)
def Json_format_check(input_str, index_line):
	# check tweet_line, which should at least be "{}"
	fflag_line = True
	if len(input_str) <= 2:
		fflag_line = False
	# check if tweet_line is complete
	fflag_json = False
	if fflag_line and input_str[0:1] == '{':
		if input_str[-2:-1] == '}' or input_str[-1:] == '}': # last line has no '\n'
			fflag_json = True
	else:
		print "Line: {}, Incomplete or Empty Line".format(index_line)
	return fflag_line, fflag_json

# single input pd.timestamp
@check_args(pd.tslib.Timestamp)
def pd_timestamp_check(input):
	return True

"""
####################################################################

Set up SQL in-RAM table variables

####################################################################
"""

def Set_TempTable_Variables(MySQL_DBkey, N_GB):
	""" 
		Parameters
		----------
		Returns
		-------
	"""	
	####################################################################
	# Connect to the database
	connection = pymysql.connect(host=MySQL_DBkey['host'],
								 user=MySQL_DBkey['user'],
								 password=MySQL_DBkey['password'],
								 db=MySQL_DBkey['db'],
								 charset=MySQL_DBkey['charset'],
								 cursorclass=pymysql.cursors.DictCursor)

	####################################################################
	comd_set_temptable = """
SET GLOBAL tmp_table_size = 1024 * 1024 * 1024 * %i;
SET GLOBAL max_heap_table_size = 1024 * 1024 * 1024 * %i;
"""
	# execute Initialize Table commands
	try:
		with connection.cursor() as cursor:
			cursor.execute( comd_set_temptable % (N_GB, N_GB) )
		# commit commands
		print "Temp table size set for 1024 * 1024 * 1024 * %i" % N_GB
		connection.commit()
	finally:
		pass
	connection.close()
	####################################################################
	return None

"""
####################################################################

Main Function of Phase2_Part1

####################################################################
"""

def Pahse2_Part1_Main(file_name_list, MySQL_DBkey, keyword1, keyword2,
					  path_save, path_tokenizer, ratio_train_test, 
					  size_dataset, thre_nonTagWords,
					  list_dict_tokenizeParameters):
	'''
		Phase2 Part1 Main Function
		tokenization and related

		Parameters
		----------
		size_dataset: total number of tweets for tokenize
		ratio_train_test: ratio of train/test
		thre_nonTagWords: the threshold for number of non-tag words, 
						  above which a tweet is selected for tokenize
		Returns
		-------
	'''

	# set in-RAM table size
	Set_TempTable_Variables(MySQL_DBkey = MySQL_DBkey, N_GB = 4)

	####################################################################

	# Connect to the database
	connection = pymysql.connect(host=MySQL_DBkey['host'],
								 user=MySQL_DBkey['user'],
								 password=MySQL_DBkey['password'],
								 db=MySQL_DBkey['db'],
								 charset=MySQL_DBkey['charset'],
								 cursorclass=pymysql.cursors.DictCursor)

	####################################################################

	# create MarkedTags_dict from .csv
	MarkedTags_dict = MarkedTag_Import(file_name_list=file_name_list)
	# extract all tweets from tweet_stack which contains key_tags
	# using MarkedTags_dict to create
	TextExtract_byTags(connection=connection, MarkedTags_dict=MarkedTags_dict, 
					   path_save=path_save, ratio_train_test=ratio_train_test,
					   keyword1=keyword1, keyword2=keyword2, size_dataset=size_dataset, 
					   thre_nonTagWords=thre_nonTagWords)

	####################################################################

	# tokenization
	for dict_case in list_dict_tokenizeParameters: 
		dict_case['N_uniqueWords'] = Tokenize_Main(dict_parameters = dict_case)
		print dict_case

	####################################################################





"""
####################################################################

# Execution of Phase2_Part1

####################################################################
"""

if __name__ == "__main__":


	####################################################################

	# ultraoct_p1v5_p2v2 data base
	# file_name_list = ['US_tweets_Oct15.txt', 'US_tweets_Oct16.txt', 'US_tweets_Oct17.txt']
	

	# ultratest_p1v5_p2v2 data base
	# file_name_list = ['US_tweets_Oct15.txt']
	# file_name_list = ['US_tweets_Oct16.txt']
	

	# ultrajuly_p1v5_p2v2
	# file_name_list = ['US_tweets_july13.txt', 'US_tweets_july15.txt', 'US_tweets_july16.txt', 
	# 					'US_tweets_july17.txt', 'US_tweets_july18.txt', 'US_tweets_july19.txt']

	####################################################################

	# MySQL_DBkey = {'host':'localhost', 'user':'sa', 'password':'fanyu01', 'db':'ultrajuly_p1v5_p2v2','charset':'utf8mb4'}
	MySQL_DBkey = {'host':'localhost', 'user':'sa', 'password':'fanyu01', 'db':'ultrajuly_p1v5_p2v2','charset':'utf8'}

	keyword1 = 'trump'
	keyword2 = 'hillary'

	####################################################################

	# tokenizer path variables
	path_preToken_DataSet = '../Data/DataSet_Tokenize/'
	path_tokenizer = './scripts/tokenizer/'

	# MarkedTag_Import file_name_list
	file_name_list = ['MarkedTag_keyword1.csv','MarkedTag_keyword2.csv']

	####################################################################

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
		'lstm_saveto': 'lstm_model_trainAgainst_trump.npz',
		'lstm_loadfrom':'lstm_model_trainAgainst_trump.npz',
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
		'lstm_saveto': 'lstm_model_trainAgainst_trump.npz',
		'lstm_loadfrom':'lstm_model_trainAgainst_trump.npz',
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

	list_dict_tokenizeParameters = [dict_tokenizeParameters_trainAgainst_trump,
									dict_tokenizeParameters_trainAgainst_hillary,
									dict_tokenizeParameters_trainAgainst_trumphillary]

	####################################################################

	Pahse2_Part1_Main(file_name_list=file_name_list, MySQL_DBkey=MySQL_DBkey, 
					  keyword1=keyword1, keyword2=keyword2, 
					  path_save=path_preToken_DataSet, path_tokenizer=path_tokenizer,
					  ratio_train_test=0.8, 
					  size_dataset=None, # total number of tweets for tokenize
					  thre_nonTagWords=10, 
					  # the threshold for number of non-tag words, 
					  # above which a tweet is selected for tokenize
					  list_dict_tokenizeParameters=list_dict_tokenizeParameters
					  # list of dicts 
					  # each dict contains parameters for tokenization for specific cases
					  # related to corresponding LSTM training
					  )

'''
####################################################################

extracting tweets from tweet_stack
total number of tweets extracted: 177033
total number of tweets scored and past tweetText filter: 112875
the ratio of train/test is: 0.800000

List of Folders involved for dictionary building:  ['posi_neut', 'posi_neg', 'neg_posi', 'neg_neut', 'neg_neg']
build_dict(data set path):  C:\Users\Nimburg\Desktop\Ultra_Phase1v5_Phase2V2\Ultra_Phase2v2\src\../Data/DataSet_Tokenize/train
Number of sentences before tokenize: 14588
Tokenizing..
Tokenizer Version 1.1
Language: en
Number of threads: 1
Done
Number of sentences after tokenize: 14588
Building dictionary..
230751  total words  19502  unique words

number of cases of posi_trump of score 1: 1452
size of training set X&Y: 1452, 1452

number of cases of neg_trump of score 0: 13136
size of training set X&Y: 14588, 14588

number of cases of posi_trump of score 1: 328
size of testing set X&Y: 328, 328

number of cases of neg_trump of score 0: 3268
size of testing set X&Y: 3596, 3596

List of Folders involved for dictionary building:  ['neut_posi', 'neg_posi', 'posi_neg', 'neut_neg', 'neg_neg']
build_dict(data set path):  C:\Users\Nimburg\Desktop\Ultra_Phase1v5_Phase2V2\Ultra_Phase2v2\src\../Data/DataSet_Tokenize/train
Number of sentences before tokenize: 11738

Building dictionary..
177728  total words  15751  unique words

number of cases of posi_hillary of score 1: 345
size of training set X&Y: 345, 345

number of cases of neg_hillary of score 0: 11393
size of training set X&Y: 11738, 11738

number of cases of posi_hillary of score 1: 79
size of testing set X&Y: 79, 79

number of cases of neg_hillary of score 0: 2843
size of testing set X&Y: 2922, 2922

List of Folders involved for dictionary building:  ['posi_neut', 'posi_neg', 'neut_posi', 'neg_posi', 'neg_neg']
build_dict(data set path):  C:\Users\Nimburg\Desktop\Ultra_Phase1v5_Phase2V2\Ultra_Phase2v2\src\../Data/DataSet_Tokenize/train
Number of sentences before tokenize: 2960

Number of sentences after tokenize: 2960
Building dictionary..
43113  total words  6723  unique words

number of cases of trump of score 2: 1432
size of training set X&Y: 1432, 1432

number of cases of hillary of score 0: 341
size of training set X&Y: 1773, 1773

number of cases of neutral of score 1: 1187
size of training set X&Y: 2960, 2960

number of cases of trump of score 2: 348
size of testing set X&Y: 348, 348

number of cases of hillary of score 0: 83
size of testing set X&Y: 431, 431

number of cases of neutral of score 1: 291
size of testing set X&Y: 722, 722




'''






