
import json
import os
import numpy as np 
import pandas as pd
import collections as col
import csv

import pymysql.cursors


'''
####################################################################
'''

def MarkedTag_Import(file_name_list):
	'''
	file_name_list: list of .csv files; 2 files, 1st for keyword 1 as trump
	'''
	# dict(), as main results structure
	# key as 'tags', value is tuple (int, int)
	# there is no separate dict() for keyword 1 & 2, as some tags over laps
	MarkedTags_dict = dict()

	counter_keyword = 0
	for file_name in file_name_list: 
		print "reading file: ", file_name
		counter_keyword += 1
		
		# read .csv file
		InputfileDir = os.path.dirname(os.path.realpath('__file__'))
		data_file_name =  '../Data/' + file_name
		Inputfilename = os.path.join(InputfileDir, data_file_name) # ../ get back to upper level
		Inputfilename = os.path.abspath(os.path.realpath(Inputfilename))
		print Inputfilename
		file_input = open(Inputfilename,'r')
		csv_data = csv.reader(file_input, delimiter=',')
		next(csv_data, None) # skip header

		# go through lines
		for row in csv_data:
			# tag, string format, ascii
			tag_text = str( row[0] ).decode('utf-8').encode('ascii', 'ignore')
			# check is tag already in MarkedTags_dict
			# alreay initialize as (0,0)
			if tag_text not in MarkedTags_dict:
				MarkedTags_dict[tag_text] = (0,0)
			# insert value
			# keyword 1
			if counter_keyword == 1:
				MarkedTags_dict[tag_text] = tuple( [ sum(x) for x in zip( MarkedTags_dict[tag_text],
																		( round( float(row[1]) ) ,0 ) 
																		) ] 
												 )
				# print tag_text, MarkedTags_dict[tag_text]
			# keyword 2
			if counter_keyword == 2:
				MarkedTags_dict[tag_text] = tuple( [ sum(x) for x in zip( MarkedTags_dict[tag_text],
																		( 0, round( float(row[1]) ) ) 
																		) ] 
												 )
				# print tag_text, MarkedTags_dict[tag_text]
	# return dict()
	return MarkedTags_dict

'''
####################################################################
'''

def TextExtract_byTags(connection, MarkedTags_dict, path_save, flag_trainOrpredict, 
					   ratio_train_test, size_dataset, 
					   thre_nonTagWords, flag_ridTags, flag_NeutralFiles, 
					   SQL_tableName, flag_alsoTxt=False):
	'''
	path_save: the path to data set before tokenize
			   '../Data/DataSet_Training/' for training LSTM
	flag_trainOrpredict: whether this is training LSTM or predicting using LSTM
						 True for training
						 False for predicting

	MarkedTags_dict: dict() of tags with scores; 
					 the set of tags based on which to extract tweets

	ratio_train_test: the ratio of train/test
	size_dataset: limit how many tweets to extract
	keyword1&2: in this case, trump and hillary

	thre_nonTagWords: the threshold of number of non-tag words
	flag_ridTags: whether to get rid of hash tags when estimating number of non-tag words
	flag_NeutralFiles: whether to write files with (0,0) scores
	
	SQL_tableName: name of the filtered_tweet_stack_'tableName'
	flag_SQLorTxt: True then output .txt file; 
				   False then only create filtered_tweet_stack

	'''
	####################################################################
	# select from tweet_stack table directly tweets that have either of the keywords
	# with the format list of lists: [tweet_id, text, taglist_str]
	Extracted_TagList = []
	# Comds for selection
	Comd_ExtractTweetText = """
SELECT tweetID, tweetText, TagList_text, tweetTime, userID
FROM tweet_stack;"""
	# Execute Comds
	print "extracting tweets from tweet_stack"
	try:
		with connection.cursor() as cursor:
			cursor.execute( Comd_ExtractTweetText )
			result = cursor.fetchall()
			# loop through all rows of this table
			counter = 0
			for entry in result:
				flag_takeTweet = False
				# print entry  
				# {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
				# in MySQL, tag format is utf8, but in raw data as ASCII
				tweetID_str = str( entry['tweetID'] ).decode('utf-8').encode('ascii', 'ignore')				
				Text_str = str( entry['tweetText'] ).decode('utf-8').encode('ascii', 'ignore')
				Text_str = Text_str.lower()
				taglist_str = str( entry['TagList_text'] ).decode('utf-8').encode('ascii', 'ignore')
				tweetTime_str = str( entry['tweetTime'] ).decode('utf-8').encode('ascii', 'ignore')
				userID_str = str( entry['userID'] ).decode('utf-8').encode('ascii', 'ignore')
				# check against MarkedTags_dict before append into Extracted_TagList
				for key in MarkedTags_dict:
					# for example, 'nevertrump' and 'nevertrumpers'
					new_key = ","+key+","
					if new_key in taglist_str:
						flag_takeTweet = True
						counter += 1
						# print "N of tweets: %i Found tag: %s" % tuple( [counter] + [key] )
						break # only need to have 1 tag from Extracted_TagList
				if flag_takeTweet == True:
					Extracted_TagList.append([tweetID_str, Text_str, taglist_str, tweetTime_str, userID_str])
	finally:
		pass
	# data check
	print "total number of tweets extracted: %i" % len(Extracted_TagList)

	####################################################################
	# split taglist_str into tags, 
	# give scores by MarkedTags_dict
	# scores are normalized
	# format: list of lists; [tweet_id_str, text_str, tuple scores by tags], encode ascii
	Extracted_Scores = []
	# go through Extracted_TagList
	for tweet in Extracted_TagList:
		tweet_score = (0,0)
		# split taglist_str into tags, update tweet_score
		taglist = tweet[2].split(',') # list[] variable
		# the flag to mark whether include this tweets or not
		# based on whether its non-tag words is above a certian threshold
		flag_nonTagWords = False 
		
		####################################################################
		# tweetText filter
		tweetText_wordList = tweet[1].split(' ')
		tweetText_new = ""
		for word in tweetText_wordList:
			# eliminate words with '@' and 'https'
			if ('@' not in word) and ('https' not in word):
				# concatenate list back to string, without @usernames
				tweetText_new += word
				tweetText_new += ' '
			# remove MenUser and https from tweetText_wordList
			# this is after concatenate this word back into string				
			if ('@' in word) or ('https' in word):
				tweetText_wordList.remove(word)
		# post filter tweetText
		tweet[1] = tweetText_new 
		# get rid of tags of this tweet; more strict thre_nonTagWords
		if flag_ridTags == True:
			for tag in taglist:
				if tag in tweetText_wordList:
					tweetText_wordList.remove(tag)
		# check against thre_nonTagWords
		if len(tweetText_wordList) >= thre_nonTagWords: 
			flag_nonTagWords = True

		####################################################################
		# pre-normalization tweet_score
		for tag in taglist:
			if tag in MarkedTags_dict:
				tweet_score = tuple( [ sum(x) for x in zip( MarkedTags_dict[tag], tweet_score ) ] 
									)
		# normalize tweet_score
		if tweet_score[0] >= 1:
			tweet_score = tuple( [ sum(x) for x in zip( (1-tweet_score[0], 0), tweet_score ) ] 
								)
		if tweet_score[0] <= -1:
			tweet_score = tuple( [ sum(x) for x in zip( (-1-tweet_score[0], 0), tweet_score ) ] 
								)
		if tweet_score[1] >= 1:
			tweet_score = tuple( [ sum(x) for x in zip( (0, 1-tweet_score[1]), tweet_score ) ] 
								)
		if tweet_score[1] <= -1:
			tweet_score = tuple( [ sum(x) for x in zip( (0, -1-tweet_score[1]), tweet_score ) ] 
								)
		
		####################################################################
		# append into Extracted_Scores
		if flag_nonTagWords == True:
			Extracted_Scores.append( [tweet[0], tweet[1], tweet_score, tweet[3], tweet[4], tweet[2] ] )
	# check
	print "total number of tweets scored and past tweetText filter: %i" % len(Extracted_Scores)
	
	####################################################################
	# load Extracted_Scores into filtered_tweet_stack

	# creating filtered_tweet_stack_'tableName'
	# Do NOT drop Tweet_Stack; Luigid consideration
	tableName = "filtered_tweet_stack_" + SQL_tableName
	Comd_TweetStack_Init = """
CREATE TABLE IF NOT EXISTS %s
(
	tweetID BIGINT PRIMARY KEY NOT NULL,
	tweetTime TIMESTAMP NOT NULL,
	userID BIGINT NOT NULL,
	tweetText varchar(3000) COLLATE utf8_bin,
	TagList_text varchar(3000), 
	Sc_Tags varchar(20)
) ENGINE=MEMORY DEFAULT CHARSET=utf8 COLLATE=utf8_bin"""
	# execute commands
	try:
		with connection.cursor() as cursor:
			cursor.execute( Comd_TweetStack_Init % tableName )
		# commit commands
		print "%s Initialized" % tableName
		connection.commit()
	finally:
		pass
	
	# load data into filtered_tweet_stack_'tableName'
	# Extracted_TagList.append([tweetID_str, Text_str, taglist_str, tweetTime_str, userID_str])
	# Extracted_Scores.append( [tweet[0], tweet[1], tweet_score, tweet[3], tweet[4], tweet[2] ] )
	print "Loading %s" % tableName
	for tweet in Extracted_Scores:
		# Comd
		comd_TweetStack_Insert = """
INSERT INTO %s (tweetID, tweetTime, userID, tweetText, TagList_text, Sc_Tags)
VALUES ( %s, '%s', %s, '%s', '%s', '%s')
ON DUPLICATE KEY UPDATE userID = %s;"""
		# execute commands
		try:
			with connection.cursor() as cursor:
				cursor.execute( comd_TweetStack_Insert % tuple( [tableName] + 
																[ tweet[0] ] + 
																[ tweet[3] ] + 
																[ tweet[4] ] + 
																[ tweet[1][:2900] ] + 
																[ tweet[5][:2900] ] + 
																[ tweet[2] ] + 
																[ tweet[4] ]
															   )
							  )
			# commit commands 
			connection.commit()
		finally:
			pass

	#Comd
	comd_convert = """
ALTER TABLE %s ENGINE=InnoDB;"""
	# execute commands
	try:
		with connection.cursor() as cursor:
			cursor.execute( comd_convert % tableName )
		# commit commands
		print tableName+" Converted"
		connection.commit()
	finally:
		pass

	print "Finished Loading %s" % tableName

	####################################################################
	# shuffle Extracted_Scores, Extracted_Scores originally in time order
	N_tweets = len(Extracted_Scores)
	idex_list =  np.arange( N_tweets )
	np.random.shuffle(idex_list)

	Extracted_Scores_temp = [ Extracted_Scores[idex] for idex in idex_list ]
	Extracted_Scores = Extracted_Scores_temp

	####################################################################
	# writing Extracted_Scores into .txt of different files
	# split extracted tweets by Extracted_Scores
	print "the ratio of train/test is: %f" % ratio_train_test
	N_tweets = len(Extracted_Scores)
	# size_dataset limit or not
	if size_dataset is not None:
		N_tweets = size_dataset
	# splitting point of train/test
	counter_train = 1.0*ratio_train_test*N_tweets
	# output dicts
	# # if for prediction, data will be in dict_train
	dict_train = {'posi_posi':[], 'posi_neut':[], 'posi_neg':[], 
				  'neut_posi':[], 'neut_neut':[], 'neut_neg':[],
				  'neg_posi':[], 'neg_neut':[], 'neg_neg':[]}

	dict_test = {'posi_posi':[], 'posi_neut':[], 'posi_neg':[], 
				 'neut_posi':[], 'neut_neut':[], 'neut_neg':[],
				 'neg_posi':[], 'neg_neut':[], 'neg_neg':[]}

	# go through Extracted_Scores
	for idx in range( N_tweets ):
		# whether this is training or predicting
		if flag_trainOrpredict == True: 
			# paths split by ratio_train_test
			if idx <= counter_train:
				path_train_test = 'train/'
			if idx > counter_train:
				path_train_test = 'test/'
		if flag_trainOrpredict == False:
			path_train_test = ''
		
		# paths split by scores
		path_score = ''
		if Extracted_Scores[idx][2][0] == 1 and Extracted_Scores[idx][2][1] == 1:
			path_score = 'posi_posi/'
		if Extracted_Scores[idx][2][0] == 1 and Extracted_Scores[idx][2][1] == 0:
			path_score = 'posi_neut/'
		if Extracted_Scores[idx][2][0] == 1 and Extracted_Scores[idx][2][1] == -1:
			path_score = 'posi_neg/'
		if Extracted_Scores[idx][2][0] == 0 and Extracted_Scores[idx][2][1] == 1:
			path_score = 'neut_posi/'
		
		if Extracted_Scores[idx][2][0] == 0 and Extracted_Scores[idx][2][1] == 0:
			# whether to write neutral files or not
			# usually too much files have (0,0) scores
			if flag_NeutralFiles == True:
				path_score = 'neut_neut/'
			elif flag_NeutralFiles == False:
				continue
		
		if Extracted_Scores[idx][2][0] == 0 and Extracted_Scores[idx][2][1] == -1:
			path_score = 'neut_neg/'
		if Extracted_Scores[idx][2][0] == -1 and Extracted_Scores[idx][2][1] == 1:
			path_score = 'neg_posi/'
		if Extracted_Scores[idx][2][0] == -1 and Extracted_Scores[idx][2][1] == 0:
			path_score = 'neg_neut/'
		if Extracted_Scores[idx][2][0] == -1 and Extracted_Scores[idx][2][1] == -1:
			path_score = 'neg_neg/'
		
		# loading into dicts
		# for training
		if path_train_test == 'train/' and path_score != '': 
			dict_train[ path_score[:-1] ].append( (Extracted_Scores[idx][0], Extracted_Scores[idx][1]) )
		if path_train_test == 'test/' and path_score != '': 
			dict_test[ path_score[:-1] ].append( (Extracted_Scores[idx][0], Extracted_Scores[idx][1]) )
		# for predicting
		if flag_trainOrpredict == False:
			# tweetID, tweetText, score_tuple by tags
			dict_train[ path_score[:-1] ].append( (Extracted_Scores[idx][0], Extracted_Scores[idx][1], Extracted_Scores[idx][2]) 
												)			

		# whether output .txt or not
		if flag_alsoTxt == True: 
			# open write close
			file_name = path_save + path_train_test + path_score + Extracted_Scores[idx][0] + '.txt'
			file = open(file_name,'w')
			file.write( Extracted_Scores[idx][1] )
			file.close()
	
	####################################################################
	# check
	for key in dict_train:
		print "dict_train check: key: %s N_sentences: %i" % tuple( [key]+[len(dict_train[key])]
																)
	for key in dict_test:
		print "dict_test check: key: %s N_sentences: %i" % tuple( [key]+[len(dict_test[key])]
																)
	# output dicts
	if flag_trainOrpredict == True:
		return dict_train, dict_test
	if flag_trainOrpredict == False:
		return dict_train, None


"""
####################################################################

Load post-Prediction Data

####################################################################
"""

def Load_Predictions(MySQL_DBkey, pred_columnName, sql_tableName, 
					 fileName_Scores_tuple, predictions_tuple):

	####################################################################
	# local function, set temp table limits
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
		return None
	####################################################################
	# set in-RAM table size
	Set_TempTable_Variables(MySQL_DBkey = MySQL_DBkey, N_GB = 4)
	# Re-Connect to the database
	connection = pymysql.connect(host=MySQL_DBkey['host'],
								 user=MySQL_DBkey['user'],
								 password=MySQL_DBkey['password'],
								 db=MySQL_DBkey['db'],
								 charset=MySQL_DBkey['charset'],
								 cursorclass=pymysql.cursors.DictCursor)

	####################################################################

	###########################################
	# Convert and Expand filtered_tweet_stack #
	###########################################

	# Comd for convert to temp table
	print "Starting convert %s to ENGINE=MEMORY" % sql_tableName
	comd_convert = """
ALTER TABLE %s ENGINE=MEMORY;"""
	# execute commands
	try:
		with connection.cursor() as cursor:
			cursor.execute( comd_convert % sql_tableName )
		# commit commands
		print sql_tableName + " Converted"
		connection.commit()
	finally:
		pass

	# Comd for adding new columns to filtered_tweet_stack
	# new columns:
	# Sc_Tags_C, pred_columnName
	# for prob_distribution and class_argmax
	pred_columnName_prob = pred_columnName + '_prob'
	pred_columnName_class = pred_columnName + '_class'
	print "Adding new columns"
	# comds
	comd_addColumns = """
ALTER TABLE %s ADD %s VARCHAR( 256 );
ALTER TABLE %s ADD %s VARCHAR( 256 );
ALTER TABLE %s ADD %s VARCHAR( 256 );"""
	print comd_addColumns % tuple( [sql_tableName] + 
													 ['Sc_Tags_C'] + 
													 [sql_tableName] + 
													 [pred_columnName_prob] +
													 [sql_tableName] + 
													 [pred_columnName_class]
								 )
	# execute commands
	try:
		with connection.cursor() as cursor:
			cursor.execute( comd_addColumns % tuple( [sql_tableName] + 
													 ['Sc_Tags_C'] + 
													 [sql_tableName] + 
													 [pred_columnName_prob] +
													 [sql_tableName] + 
													 [pred_columnName_class]
													)
						  )
		# commit commands
		connection.commit()
	finally:
		pass

	#####################################################
	# Add predictions to filtered_tweet_stack then Save #
	#####################################################

	# extract list variables from tuple_of_list varialbes
	# fileName_Scores_tuple = ( fileNames, scores_tag )
	# predictions_tuple = ( dataset_preds_prob, dataset_preds)
	fileNames = fileName_Scores_tuple[0]
	scores_tag = fileName_Scores_tuple[1]
	dataset_preds_prob = predictions_tuple[0]
	dataset_preds = predictions_tuple[1]
	# locate the min_len() of these variables, raise Error if not same
	len_list = [len(fileNames), len(scores_tag), len(dataset_preds_prob), len(dataset_preds)]
	idx_limit = min( len_list )
	print "idx_limit, ", idx_limit

	# loop through
	for idx in range(idx_limit):

		# comd_Insert variables; all string format
		str_tweetID = str( fileNames[idx] )
		str_Sc_Tags = str( scores_tag[idx] )
		
		if len(dataset_preds_prob[idx,:]) == 2:
			str_prob = "(%.2f, %.2f)" % tuple( [ dataset_preds_prob[idx,0] ] + 
											   [ dataset_preds_prob[idx,1] ]
											 )
		
		if len(dataset_preds_prob[idx,:]) == 3:
			str_prob = "(%.2f, %.2f, %.2f)" % tuple( [ dataset_preds_prob[idx,0] ] + 
											   		 [ dataset_preds_prob[idx,1] ] + 
											   		 [ dataset_preds_prob[idx,2] ]
											 		)

		str_class = "%.1f" % round( dataset_preds[idx], 1)
		# command to insert Sc_Tags_C, pred_columnName_prob, pred_columnName_class
		comd_Update = """
UPDATE %s 
SET Sc_Tags_C='%s', %s='%s', %s='%s'
WHERE tweetID = %s;"""
		# execute commands
		try:
			with connection.cursor() as cursor:
				cursor.execute( comd_Update % tuple( [sql_tableName] + 
													 [str_Sc_Tags] + 
													 [pred_columnName_prob] + 
													 [str_prob] + 
													 [pred_columnName_class] + 
													 [str_class] + 
													 [str_tweetID]
													) 
							  )
			# commit commands
			connection.commit()
		finally:
			pass

	# Comd for convert to temp table
	print "Starting convert %s to ENGINE=InnoDB" % sql_tableName
	comd_convert = """
ALTER TABLE %s ENGINE=InnoDB;"""
	# execute commands
	try:
		with connection.cursor() as cursor:
			cursor.execute( comd_convert % sql_tableName )
		# commit commands
		print sql_tableName + " Converted"
		connection.commit()
	finally:
		pass

	####################################################################
	return None


"""
####################################################################

test codes

####################################################################
"""

if __name__ == "__main__":

	file_name_list = ['Test_MarkedTag_Key1.csv','Test_MarkedTag_Key2.csv']
	path_preToken_DataSet = '../Data/DataSet_Tokenize/'
	

	# MarkedTags_dict = MarkedTag_Import(file_name_list=file_name_list)



