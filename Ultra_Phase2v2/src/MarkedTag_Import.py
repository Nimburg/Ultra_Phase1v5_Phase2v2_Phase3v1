
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
																		( int(row[1]) ,0 ) ) ] 
												 )
				# print tag_text, MarkedTags_dict[tag_text]
			# keyword 2
			if counter_keyword == 2:
				MarkedTags_dict[tag_text] = tuple( [ sum(x) for x in zip( MarkedTags_dict[tag_text],
																		( 0, int(row[1]) ) ) ] 
												 )
				# print tag_text, MarkedTags_dict[tag_text]
	# return dict()
	return MarkedTags_dict

'''
####################################################################
'''

def TextExtract_byTags(connection, MarkedTags_dict, path_save, ratio_train_test, size_dataset, 
					   keyword1, keyword2, thre_nonTagWords):
	'''
	path_save: the path to data set before tokenize
	ratio_train_test: the ratio of train/test
	'''
	####################################################################
	# select from tweet_stack table directly tweets that have either of the keywords
	# with the format list of lists: [tweet_id, text, taglist_str]
	Extracted_TagList = []
	# Comds for selection
	Comd_ExtractTweetText = """
SELECT tweetID, tweetText, TagList_text
FROM tweet_stack
where TagList_text LIKE '%s' or TagList_text LIKE '%s';"""
	keyword1_sql = '%'+keyword1+'%'
	keyword2_sql = '%'+keyword2+'%'
	# Execute Comds
	print "extracting tweets from tweet_stack"
	try:
		with connection.cursor() as cursor:
			cursor.execute( Comd_ExtractTweetText % tuple( [keyword1_sql]+[keyword2_sql] ) )
			result = cursor.fetchall()
			# loop through all rows of this table
			for entry in result:
				# print entry  
				# {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
				# in MySQL, tag format is utf8, but in raw data as ASCII
				tweetID_str = str(entry['tweetID']).decode('utf-8').encode('ascii', 'ignore')
				Text_str = str(entry['tweetText']).decode('utf-8').encode('ascii', 'ignore')
				Text_str = Text_str.lower()
				taglist_str = str(entry['TagList_text']).decode('utf-8').encode('ascii', 'ignore')
				Extracted_TagList.append([tweetID_str, Text_str, taglist_str])
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
			# remove MenUser names from tweetText_wordList
			# this is after concatenate this word back into string			
			# check against keyword 1&2, against MenUsers and https		
			if ('@' in word) or ('https' in word) or (keyword1 in word) or (keyword2 in word):
				tweetText_wordList.remove(word)
		# post filter tweetText
		tweet[1] = tweetText_new		
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
			Extracted_Scores.append( [tweet[0], tweet[1], tweet_score ] )
	# check
	print "total number of tweets scored and past tweetText filter: %i" % len(Extracted_Scores)
	
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
	# go through Extracted_Scores
	for idx in range( N_tweets ):
		# paths split by ratio_train_test
		if idx <= counter_train:
			path_train_test = 'train/'
		if idx > counter_train:
			path_train_test = 'test/'
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
			#path_score = 'neut_neut/'
			continue
		
		if Extracted_Scores[idx][2][0] == 0 and Extracted_Scores[idx][2][1] == -1:
			path_score = 'neut_neg/'
		if Extracted_Scores[idx][2][0] == -1 and Extracted_Scores[idx][2][1] == 1:
			path_score = 'neg_posi/'
		if Extracted_Scores[idx][2][0] == -1 and Extracted_Scores[idx][2][1] == 0:
			path_score = 'neg_neut/'
		if Extracted_Scores[idx][2][0] == -1 and Extracted_Scores[idx][2][1] == -1:
			path_score = 'neg_neg/'
		# open write close
		file_name = path_save + path_train_test + path_score + Extracted_Scores[idx][0] + '.txt'
		file = open(file_name,'w')
		file.write( Extracted_Scores[idx][1] )
		file.close()
	
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
	


	MarkedTags_dict = MarkedTag_Import(file_name_list=file_name_list)
























