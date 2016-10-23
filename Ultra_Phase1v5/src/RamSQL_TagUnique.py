import json
import os
import numpy as np 
import pandas as pd
import collections as col


####################################################################
# intended for two keyword (2 projection axis)
class RamSQL_Tag:

	def __init__(self, tagText, tweetID, user, ):
		
		# basic items of initialization
		self.tagText = tagText
		self.totalCall = 1 # totalCall should match length of tweetID_list

		# items for keyword 1; 
		# length of tagScore[] should match length of tagNcall
		# length of tagScore[] and tagNcall[] may different from totalCall
		self.tagScore1 = []
		self.tagNcall1 = []

		# items for keyword 2; length of tagScore[] should match length of tagNcall
		self.tagScore2 = []
		self.tagNcall2 = []

		# extended item for both keywords
		self.tweetID_list = []
		self.user_counter = col.Counter() # use [key] += 1 directly
		self.tag_counter = col.Counter()

		# update tweetID and tweet author
		if tweetID.isdigit() == True:
			self.tweetID_list.append(tweetID)
		if user.isdigit() == True:
			self.user_counter[user] += 1

	####################################################################

	# should have datatype check in the main program body
	def update_N_score(self, tweetID=None, user=None, tagCon=None):

		# update tweetID_list and totalCall
		# need to further guarantee in main body program that this is NOT updated twice
		if tweetID != None and tweetID.isdigit() == True:
			self.tweetID_list.append(tweetID)
			self.totalCall += 1

		# update reply and mentionuser
		# need to further guarantee in main body program that this user is NOT tweet author
		if user != None and user.isdigit() == True:
			self.user_counter[user] += 1

		# update tag
		if tagCon != None and tagCon != self.tagText:
			self.tag_counter[tagCon] += 1

	# both scores needed to be updated together
	def update_score(self, score1=0, Ncall1=0, score2=0, Ncall2=0):
		# update score1 and score2
		if score1 != 0 and Ncall1 != 0:
			self.tagScore1.append(score1)
			self.tagNcall1.append(Ncall1)
		else:
			self.tagScore1.append(0)
			self.tagNcall1.append(0)

		if score2 != 0 and Ncall2 != 0:
			self.tagScore2.append(score2)
			self.tagNcall2.append(Ncall2)
		else:
			self.tagScore2.append(0)
			self.tagNcall2.append(0)

	####################################################################
	# parse list variables into string

	# parse tweetID_list into string
	def tweetID_list_str(self):
		list_str = ""
		for item in self.tweetID_list:
			list_str = list_str + item + ","
		if len(list_str) > 1:
			list_str = list_str[:-1] # get rid of the last ','
		return list_str

	# parse tagScore1
	def tagScore1_str(self):
		list_str = ""
		for item in self.tagScore1:
			list_str = list_str +"%.3f,"%round(item,3)
		if len(list_str) > 1:
			list_str = list_str[:-1] # get rid of the last ','
		return list_str


	# parse tagNcall1
	def tagNcall1_str(self):
		list_str = ""
		for item in self.tagNcall1:
			list_str = list_str + str(item) + ","
		if len(list_str) > 1:
			list_str = list_str[:-1] # get rid of the last ','
		return list_str

	# parse tagScore2
	def tagScore2_str(self):
		list_str = ""
		for item in self.tagScore2:
			list_str = list_str + "%.3f,"%round(item,3)
		if len(list_str) > 1:
			list_str = list_str[:-1] # get rid of the last ','
		return list_str

	# parse tagNcall1
	def tagNcall2_str(self):
		list_str = ""
		for item in self.tagNcall2:
			list_str = list_str + str(item) + ","
		if len(list_str) > 1:
			list_str = list_str[:-1] # get rid of the last ','
		return list_str

	####################################################################
	# parse dict variables into JSON

	# parse tag_counter
	def tag_counter_str(self):
		dict_str = ""
		for key in self.tag_counter:
			dict_str = dict_str+key+":"+str(self.tag_counter[key])+","
		if len(dict_str) > 1:
			dict_str = dict_str[:-1]
		return dict_str

	# parse user_counter
	def user_counter_str(self):
		dict_str = ""
		for key in self.user_counter:
			dict_str = dict_str+key+":"+str(self.user_counter[key])+","
		if len(dict_str) > 1:
			dict_str = dict_str[:-1]
		return dict_str

	####################################################################
	# just in case
	def selfPrint(self):
		print self.tagText, ",	total call: ", self.totalCall, ",	len of tweetID_list: ", len(self.tweetID_list)
		print "tweetID_list: ", self.tweetID_list_str()
		
		print "score of keyword1: ", self.tagScore1_str()
		print "keyword1 Ncall: ", self.tagNcall1_str()
		print "score of keyword2: ", self.tagScore2_str()
		print "keyword2 Ncall: ", self.tagNcall2_str()
		
		print "user_counter", self.user_counter_str()

		print "tag_counter", self.tag_counter_str()


"""
#####################################################################################

test code

#####################################################################################
"""

if __name__ == "__main__":

	RamSQL_TagUnique = col.defaultdict(RamSQL_Tag)

	####################################################################

	RamSQL_TagUnique['trump'] = RamSQL_Tag(tagText='trump', tweetID='12345', user='123')
	RamSQL_TagUnique['trump'].update_N_score(tweetID='12346', user='234')
	RamSQL_TagUnique['trump'].update_N_score(tweetID='12346', user='345', tagCon='trump')
	RamSQL_TagUnique['trump'].update_N_score(tweetID='12346', user='456', tagCon='hillary')
	RamSQL_TagUnique['trump'].update_score(score1 = 8.9, Ncall1=1)

	RamSQL_TagUnique['trump'].selfPrint()








