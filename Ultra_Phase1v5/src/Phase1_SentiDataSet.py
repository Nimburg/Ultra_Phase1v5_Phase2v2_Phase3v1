
####################################################################
# 
# After Phase1_Main, one should:
# 
# 1. extract all tag_key1&2 from the latest set of RollingScoreBank 
# and export such set for manual marking
# into .csv format
# each tag to be set values (int, int)
# keyword1 = 'trump' keyword2 = 'hillary'
# +1: support, -1: against 0: neutral or irrelevent
# 
# 
# 
# 
# 
# 
# 
# 2. import the marked tags
# from .csv format
# 
# 3. use the marked tags to export tweet text as training/testing data sets
# 
# 
# 
# 
# tokenize and pickle data???
# 
# 
####################################################################


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

def KeyTags_Extract(MySQL_DBkey, str_pin_time):

	####################################################################
	# Connect to the database
	connection = pymysql.connect(host=MySQL_DBkey['host'],
								 user=MySQL_DBkey['user'],
								 password=MySQL_DBkey['password'],
								 db=MySQL_DBkey['db'],
								 charset=MySQL_DBkey['charset'],
								 cursorclass=pymysql.cursors.DictCursor)
	####################################################################	

	# name of key tag tables
	table_tag_key1 = "RSB_tag_key1_"+str_pin_time
	table_tag_key2 = "RSB_tag_key2_"+str_pin_time

	# Comds for selecting data
	# table_tag_key1 &2
	comd_tag_key = """
select tag, tag_Ncall
from %s"""
	
	# execute command
	# load table_tag_key1 & 2
	counter = 0
	for table_Name in [ table_tag_key1, table_tag_key2]:
		counter += 1
		KeyTags_list = []
		if counter == 1:
			csv_fileName = 'tag_keyword1.csv'
		elif counter == 2:
			csv_fileName = 'tag_keyword2.csv'
		try:
			with connection.cursor() as cursor:
				cursor.execute( comd_tag_key%table_Name )
				result = cursor.fetchall()
				# loop through all rows of this table
				for entry in result:		
					KeyTags_list.append([str(entry['tag']), int(entry['tag_Ncall'])])
		finally:
			pass	
		# output csv
		OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
		data_file_name =  '../Outputs/' + csv_fileName
		Outputfilename = os.path.join(OutputfileDir, data_file_name) # ../ get back to upper level
		Outputfilename = os.path.abspath(os.path.realpath(Outputfilename))
		print Outputfilename
		
		KeyTags_pd = pd.DataFrame(KeyTags_list)
		KeyTags_pd.to_csv(Outputfilename, index=False, header=False)

	return None

'''
####################################################################
'''
















