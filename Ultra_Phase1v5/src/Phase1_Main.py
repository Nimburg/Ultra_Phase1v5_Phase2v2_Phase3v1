
####################################################################
# 
# Design
# 
# Phase1_Main will perform readline loop control, handling each tweet
# define variables: RollingScoreBank, Tweet_OBJ, RamSQL and several flags
# it handles data in a per-tweet fashion
# 
# Stage0 will read JSON tweet, check conditions, (if) extract all infor into Tweet_OBJ, update pin_time
# (if) pin_time, initialize DB tables
# 
# Stage1 will update RollingScoreBank; 
# This is mostly a pure in-RAM operation;
# After each window, Backing-up into the format of in-RAM SQL table; then Backing-up into SQL data base;
# 
# Stage3 will Define and Start updating RamSQL;
# This is pure in-RAM operation
# 
####################################################################
#
# RollingScoreBank is save into 6 tables:
# 
# key tags related to 'trump' and its N_call
# key tags related to 'hillary' and its N_call
# 
# relevant tags related to 'trump' and its score
# relevant tags related to 'trump' and its N_call
# 
# relevant tags related to 'hillary' and its score
# relevant tags related to 'hillary' and its N_call
#
# users related to 'trump' and its score
# users related to 'trump' and its N_call
# 
# users related to 'hillary' and its score
# users related to 'hillary' and its N_call
#
####################################################################
# 
####################################################################
# 
# After Phase1_Main, one should:
# 1. extract all tag_key1&2 from the latest set of RollingScoreBank
# 2. export such set for manual marking
# 
####################################################################


import json
import os
import numpy as np 
import pandas as pd
import collections as col

import pymysql.cursors

from RamSQL_TagUnique import RamSQL_Tag
from RamSQL_UserUnique import RamSQL_User

from Phase1_Stage0 import Stage0_Json
from Phase1_Stage0_TablePrep import TweetStack_Init, TweetStack_load
from Phase1_Stage0_TablePrep import TagUnique_Init, TagUnique_Insert, TagUnique_Convert
from Phase1_Stage0_TablePrep import UserUnique_Init, UserUnique_Insert, UserUnique_Convert
from Phase1_Stage0_TablePrep import RollingScoreBank_Save, RollingScoreBank_Load

from Phase1_Stage1 import RollingScore_Update
from Phase1_Stage3 import RamSQL_UserUnique_update, RamSQL_TagUnique_update

from Phase1_SentiDataSet import KeyTags_Extract


"""
####################################################################
"""

####################################################################
# variable type check
def check_args(*types):
	def real_decorator(func):
		def wrapper(*args, **kwargs):
			for val, typ in zip(args, types):
				assert isinstance(val, typ), "Value {} is not of expected type {}".format(val, typ)
			return func(*args, **kwargs)
		return wrapper
	return real_decorator

####################################################################

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

####################################################################

def DataFile_Iter(file_name_list, file_idex):
	'''
	go through file_name_list
	returns file_input = open(Inputfilename,'r') of each data file
	'''
	# read tweets.txt file data
	InputfileDir = os.path.dirname(os.path.realpath('__file__'))
	print InputfileDir
	data_file_name =  '../Data/' + file_name_list[file_idex]
	Inputfilename = os.path.join(InputfileDir, data_file_name) # ../ get back to upper level
	Inputfilename = os.path.abspath(os.path.realpath(Inputfilename))
	print Inputfilename
	file_input = open(Inputfilename,'r')

	return file_input

####################################################################

def Phase1_RollingScoreBank_load(MySQL_DBkey, RollingScoreBank, str_pin_time):

	####################################################################
	# Connect to the database
	connection = pymysql.connect(host=MySQL_DBkey['host'],
								 user=MySQL_DBkey['user'],
								 password=MySQL_DBkey['password'],
								 db=MySQL_DBkey['db'],
								 charset=MySQL_DBkey['charset'],
								 cursorclass=pymysql.cursors.DictCursor)

	####################################################################
	RollingScoreBank = RollingScoreBank_Load(connection=connection, 
											 RollingScoreBank=RollingScoreBank, 
											 str_pin_time=str_pin_time)

	return RollingScoreBank

####################################################################

"""
####################################################################

Phase1 Main Function

####################################################################
"""


def Phase1_Main(file_name_list, MySQL_DBkey, time_WindowSize,
				RollingScoreBank, keyword1, keyword2, 
				thre_tag_mid, thre_tag_low, thre_tag_reset,
				thre_user_mid, thre_user_low, thre_user_reset):
	""" 
		Phase 1 Main Function

		Parameters
		----------
		MySQL_DBkey: 
		file_name: 
		keyword1: 
		keyword2:
		time_WindowSize: how wide is each window per second

		Returns
		-------
		RollingScoreBank
	"""	

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

	# main data structure, contains information from each tweet
	Tweet_OBJ = col.defaultdict(set)
	# initialize
	Tweet_OBJ['tweet_time'] = set([]) # pd timestamp
	Tweet_OBJ['tweet_id'] = set([]) # all id are id_str
	Tweet_OBJ['user_id'] = set([])
	Tweet_OBJ['user_name'] = set([])
	Tweet_OBJ['user_followers'] = set([])
	Tweet_OBJ['user_friends'] = set([])
	# eliminate reapeating tags
	# the following three will be: K & R, only R, only due_user
	Tweet_OBJ['Tag_Keyword'] = set([])
	Tweet_OBJ['Tag_Relevant'] = set([])
	Tweet_OBJ['Tag_due_user'] = set([])
	# make sure text is onto a single line.....
	Tweet_OBJ['text'] = set([])
	# eliminate repeating userIDs
	Tweet_OBJ['reply_to_userID'] = set([])
	Tweet_OBJ['mentioned_userID'] = set([])

	####################################################################
	# create table TweetStack
	TweetStack_Init(connection=connection)
	# for 1st sliding window
	# RamSQL_TagUnique
	RamSQL_TagUnique = col.defaultdict(RamSQL_Tag)
	# RamSQL_UserUnique
	RamSQL_UserUnique = col.defaultdict(RamSQL_User)
	####################################################################
	
	# main logic structure, controled by readline() returns, exist at end of file
	flag_fileEnd = True # to kick start the 1st file
	count_emptyline = 0
	count_line = 0
	pin_time = pd.to_datetime("Thu Oct 29 18:51:50 +0000 2015") # make sure it is old enough

	# this is the logic loop for EACH tweet
	file_index = 0
	file_Ntotal = len(file_name_list)

	while ( file_index < file_Ntotal ):
		# for the start of each file
		if flag_fileEnd == True:
			flag_fileEnd = False
			file_input = DataFile_Iter(file_name_list=file_name_list, file_idex=file_index)
			count_line = 0
		
		count_line += 1 
		tweet_line = file_input.readline()
		
		# json format check and file end check and data type check
		flag_line = False
		flag_json = False
		try:
			flag_line, flag_json = Json_format_check(tweet_line, count_line)
		except AssertionError:
			print "Line: {}, Json_format_check() data type check failed".format(index_line)
			pass 
		else:
			pass
		# count adjacent empty lines
		if flag_line == True:
			count_emptyline = 0
		else:
			count_emptyline += 1
		# flag end of file
		if count_emptyline > 4:
			flag_fileEnd = True
			file_index += 1

		####################################################################
		# Stage0
		# read JSON tweet, check conditions, (if) extract all infor into Tweet_OBJ, update pin_time
		# input/output RollingScoreBank; output Tweet_OBJ
		flag_TweetStack = False
		if flag_json == True:		
			
			# load JSON, extract information
			flag_TweetStack, Tweet_OBJ = Stage0_Json(input_str=tweet_line, index_line=count_line, 
				keyword1=keyword1, keyword2=keyword2, RollingScoreBank=RollingScoreBank,
				thre_tag_mid=thre_tag_mid, thre_tag_low=thre_tag_low, thre_tag_reset=thre_tag_reset,
				thre_user_mid=thre_user_mid, thre_user_low=thre_user_low, thre_user_reset=thre_user_reset)

			# if JSON extraction successful, check pin_time, check sliding_window
			flag_timeStamp = False
			if flag_TweetStack:
				# DataTypeCheck for pd.timestamp; compare and update pin_time
				try:
					flag_timeStamp = pd_timestamp_check(next(iter(Tweet_OBJ['tweet_time'])))
				except AssertionError:
					print "pin_time datatype failed"
					pass 
				else:
					pass

			# check pin_time
			Delta_time = 0
			flag_new_window = False
			if flag_TweetStack and flag_timeStamp:
				Delta_time = (next(iter(Tweet_OBJ['tweet_time'])) - pin_time)/np.timedelta64(1,'s')
			
			# cue for creating
			if Delta_time > time_WindowSize and Delta_time < 86400: # one day
				flag_new_window = True
				pin_time_load = pin_time
				pin_time = next(iter(Tweet_OBJ['tweet_time']))

			# create TagUnique and UserUnique DB tables for the 1st sliding_window
			# create RamSQL variables for 1st sliding window
			# create 1st RamSQL
			if Delta_time >= 86400:
				pin_time = next(iter(Tweet_OBJ['tweet_time']))
				# Initialize DB table TagUnique
				TagUnique_Init(connection=connection, pin_time=pin_time)
				# Initialize DB table UserUnique
				UserUnique_Init(connection=connection, pin_time=pin_time)


		####################################################################
		# Stage1
		# update RollingScoreBank
		if flag_TweetStack:
			RollingScoreBank = RollingScore_Update(RollingScoreBank=RollingScoreBank, 
												   Tweet_OBJ=Tweet_OBJ, keyword1=keyword1, keyword2=keyword2,
												   thre_tag_mid=thre_tag_mid, thre_tag_low=thre_tag_low, thre_tag_reset=thre_tag_reset,
												   thre_user_mid=thre_user_mid, thre_user_low=thre_user_low, thre_user_reset=thre_user_reset)

		####################################################################
		# Stage2
		# DataType Check
		# upload TweetStack
		if flag_TweetStack:
			TweetStack_load(connection=connection, Tweet_OBJ=Tweet_OBJ)

		####################################################################
		# Stage3
		# DataType Check
		# updating RamSQL
		if flag_TweetStack:
			# update RamSQL_TagUnique
			RamSQL_TagUnique = RamSQL_TagUnique_update(RamSQL_TagUnique=RamSQL_TagUnique,
				Tweet_OBJ=Tweet_OBJ, RollingScoreBank=RollingScoreBank)
			# update RamSQL_UserUnique
			RamSQL_UserUnique = RamSQL_UserUnique_update(RamSQL_UserUnique=RamSQL_UserUnique, 
				Tweet_OBJ=Tweet_OBJ, RollingScoreBank=RollingScoreBank)

		####################################################################
		# Stage4
		# (if) pin_time or (if) end_of_file, load RamSQL into DB SQL, signal for new sliding window
		if flag_TweetStack and flag_new_window:
			# load RamsQL_TagUnique into DB table TagUnique, of existing sliding window
			print "TagUnique_Insert"
			TagUnique_Insert(connection=connection, pin_time=pin_time_load, RamSQL_TagUnique=RamSQL_TagUnique)
			# load RamsQL_UserUnique into DB table TagUnique, of existing sliding window
			print "UserUnique_Insert"
			UserUnique_Insert(connection=connection, pin_time=pin_time_load, RamSQL_UserUnique=RamSQL_UserUnique)
			# convert old sliding window in-RAM tabes into in-disk tables
			print "converting in-RAM tables"
			TagUnique_Convert(connection=connection, pin_time=pin_time_load)
			UserUnique_Convert(connection=connection, pin_time=pin_time_load)
			# save RollingScoreBank
			RollingScoreBank_Save(connection=connection, pin_time=pin_time_load, RollingScoreBank=RollingScoreBank)

		# for the last window of the last file
		if flag_fileEnd == True and file_index == file_Ntotal:
			# load RamsQL_TagUnique into DB table TagUnique, of existing sliding window
			print "TagUnique_Insert"
			TagUnique_Insert(connection=connection, pin_time=pin_time, RamSQL_TagUnique=RamSQL_TagUnique)
			# load RamsQL_UserUnique into DB table TagUnique, of existing sliding window
			print "UserUnique_Insert"
			UserUnique_Insert(connection=connection, pin_time=pin_time, RamSQL_UserUnique=RamSQL_UserUnique)
			# convert old sliding window in-RAM tabes into in-disk tables
			print "converting in-RAM tables"
			TagUnique_Convert(connection=connection, pin_time=pin_time)
			UserUnique_Convert(connection=connection, pin_time=pin_time)
			# save RollingScoreBank
			RollingScoreBank_Save(connection=connection, pin_time=pin_time, RollingScoreBank=RollingScoreBank)

		# create New DB table TagUnique and UserUnique
		# create New RamSQL
		if flag_TweetStack and flag_new_window:
			# operations ofr creating a new sliding window
			print "create new sliding window "+pin_time.strftime('%Y-%m-%d-%H')
			# create New DB table TagUnique
			TagUnique_Init(connection=connection, pin_time=pin_time)
			# NEW RamSQL_TagUnique
			RamSQL_TagUnique = col.defaultdict(RamSQL_Tag)
			# create New DB table UserUnique
			UserUnique_Init(connection=connection, pin_time=pin_time)
			# NEW RamSQL_UserUnique
			RamSQL_UserUnique = col.defaultdict(RamSQL_User)


	####################################################################
	# End of File
	connection.close()
	return RollingScoreBank


"""
####################################################################

# Execution of Phase1

####################################################################
"""

if __name__ == "__main__":

	# controls whether load existing RollingScoreBank
	flag_load_RollingScoreBank = False  
	str_pin_time_RSB = '2016_07_27_09' # what time pin

	# controls Phase1_Main(); whether loading data files and creating/extending data base
	flag_Phase1_Main = False  

	# controls whether to etract tag_key1&2 from the latest set of RollingScoreBank after Phase1_Main()
	flag_tagExtract = False 
	str_pin_time_tag = '2016_07_27_09'

	####################################################################

	# ultraoct_p1v5_p2v2 data base
	# file_name_list = ['US_tweets_Oct15.txt', 'US_tweets_Oct16.txt', 'US_tweets_Oct17.txt']
	

	# ultratest_p1v5_p2v2 data base
	# file_name_list = ['US_tweets_Oct15.txt']
	# file_name_list = ['US_tweets_Oct16.txt']
	

	# ultrajuly_p1v5_p2v2
	# file_name_list = ['US_tweets_july13.txt', 'US_tweets_july15.txt', 'US_tweets_july16.txt', 
	# 					'US_tweets_july17.txt', 'US_tweets_july18.txt', 'US_tweets_july19.txt']
	# file_name_list = ['US_tweets_july20.txt', 'US_tweets_july22.txt', 'US_tweets_july23.txt', 
	# 					'US_tweets_july24.txt', 'US_tweets_july25.txt', 'US_tweets_july26.txt']
	file_name_list = ['US_tweets_july27.txt', 'US_tweets_july29.txt', 'US_tweets_july30.txt', 'US_tweets_july31.txt', 'US_tweets_Aug1.txt', 'US_tweets_Aug2.txt']

	####################################################################

	# MySQL_DBkey = {'host':'localhost', 'user':'sa', 'password':'fanyu01', 'db':'ultrajuly_p1v5_p2v2','charset':'utf8mb4'}
	MySQL_DBkey = {'host':'localhost', 'user':'sa', 'password':'fanyu01', 'db':'ultrajuly_p1v5_p2v2','charset':'utf8'}

	keyword1 = 'trump'
	keyword2 = 'hillary'

	####################################################################

	# Rolling Score Bank
	# This variable is "global" across ALL data files; 
	# this variable does NOT got wiped with each sliding window
	# Rolling Score Bank should be checked to create flags, rather than directly control RamSQL
	RollingScoreBank = col.defaultdict(col.Counter)
	# tags that contain keywords
	RollingScoreBank['tag_keyword1'] = col.Counter() # val = N_call
	RollingScoreBank['tag_keyword2'] = col.Counter() # val = N_call
	# tags that with score >= 5
	# keys will overlap here
	RollingScoreBank['tag_relevant1'] = col.Counter() # val = score
	RollingScoreBank['tag_relevant1_N'] = col.Counter() # val = N_call
	RollingScoreBank['tag_relevant2'] = col.Counter() # val = score
	RollingScoreBank['tag_relevant2_N'] = col.Counter() # val = N_call
	# list of users
	RollingScoreBank['user1'] = col.Counter() # key = id_str, val = score
	RollingScoreBank['user1_N'] = col.Counter() # key = id_str, val = N_act
	RollingScoreBank['user2'] = col.Counter() # key = id_str, val = score
	RollingScoreBank['user2_N'] = col.Counter() # key = id_str, val = N_act

	####################################################################	
	# load existing RollingScoreBank, if there is one
	if flag_load_RollingScoreBank == True:
		RollingScoreBank = Phase1_RollingScoreBank_load(MySQL_DBkey=MySQL_DBkey, 
														RollingScoreBank=RollingScoreBank, 
														str_pin_time=str_pin_time_RSB)

	####################################################################
	# execute Phase1_Main function
	# through file_name_list, create the SQL data base
	
	if flag_Phase1_Main == True:
		RollingScoreBank = Phase1_Main(file_name_list = file_name_list, keyword1 = keyword1, keyword2 = keyword2, 
									   MySQL_DBkey = MySQL_DBkey, RollingScoreBank = RollingScoreBank, 
									   time_WindowSize = 4*3600, # 
									   thre_tag_mid=5.0, thre_tag_low=1.0, thre_tag_reset=2.0,
									   thre_user_mid=1.0, thre_user_low=0.1, thre_user_reset=1.2
									   )

		print "End of Execution of Phase 1"

	####################################################################
	# extract all tag_key1&2 from the latest set of RollingScoreBank 
	# and export such set for manual marking
	if flag_tagExtract == True:
		KeyTags_Extract(MySQL_DBkey=MySQL_DBkey, str_pin_time=str_pin_time_tag)
















