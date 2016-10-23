import json
import os
import numpy as np 
import pandas as pd
import collections as col

import pymysql.cursors

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



####################################################################
# parse multi-line string noto single line
def parse_MultiLine_text(input_str):
	temp_list = input_str.splitlines()
	res = ""
	for item in temp_list:
		res += item + ' '
	return res

# force json.loads() take it as ASCii
def transASC(input): 
    if isinstance(input, dict):
    	tempdic = dict()
    	for key,value in input.iteritems():
    		tempdic[transASC(key)] = transASC(value)
    	return tempdic
    elif isinstance(input, list):
        return [transASC(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

# get ride of utf-8
def removeUtf(text_input):
	listtext = list(text_input)
	for j in range(len(listtext)): # handle utf-8 issue
		try:
			listtext[j].encode('utf-8')
		except UnicodeDecodeError:
			listtext[j] = ''
		if listtext[j] == '\n': # for those with multiple line text
			listtext[j] = ''
	text_input = ''.join(listtext)
	return text_input


"""
####################################################################
"""

def Stage0_Json(input_str, index_line, keyword1, keyword2, 
				RollingScoreBank, 
				thre_tag_mid, thre_tag_low, thre_tag_reset,
				thre_user_mid, thre_user_low, thre_user_reset):

	####################################################################

	# main data structure, contains information from each tweet
	# to be returned
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

	#############################################################
	# json load, extract tweet time and id_str
	flag_TidTimeAuthor = True # flag for tweet id, time and author
	try:
		# load json
		tweet_json = json.loads(input_str)
	except ValueError: 
		print "Line: {}, json loads Error".format(index_line)
		flag_TidTimeAuthor = False
	else:	
		# extract date-time from mainbody
		try:
			time_str = tweet_json['created_at']
			tweet_id = tweet_json['id_str']
		except ValueError:
			flag_TidTimeAuthor = False
			pass
		except KeyError:
			flag_TidTimeAuthor = False
			pass
		else:
			# convert to pandas timestamp
			try:
				time_dt = pd.to_datetime(time_str)
				if pd.isnull(time_dt):
					flag_TidTimeAuthor = False
					print "Line: {}, date-time is NaT".format(index_line)
			except ValueError:
				flag_TidTimeAuthor = False
				print "Line: {}, date-time convertion failed".format(index_line)
				pass
			else:
				# upload to RetD_TimeUserTag
				if flag_TidTimeAuthor:
					Tweet_OBJ['tweet_time'] = set([time_dt])
					Tweet_OBJ['tweet_id'] = set([tweet_id])

	#############################################################
	# extract user information sub-json
	if flag_TidTimeAuthor:
		try:
			user_json = tweet_json['user']
		except ValueError:
			flag_TidTimeAuthor = False
			pass
		except KeyError:
			flag_TidTimeAuthor = False
			pass
		else:
			# extract user statistics
			try: 
				user_id = user_json['id_str']
				user_name = user_json['screen_name']
				if len(user_name) > 253:
					user_name = user_name[:250]
				user_followers = user_json['followers_count']
				user_friends = user_json['friends_count']
			except ValueError:
				flag_TidTimeAuthor = False
				pass
			except KeyError:
				flag_TidTimeAuthor = False
				pass
			else:
				if flag_TidTimeAuthor:
					Tweet_OBJ['user_id'] = set([user_id])
					Tweet_OBJ['user_name'] = set([user_name])
					Tweet_OBJ['user_followers'] = set([user_followers])
					Tweet_OBJ['user_friends'] = set([user_friends])
					# extract reply_to_user information
					try:
						reply_userID_str = tweet_json['in_reply_to_user_id_str']
						# if userID == null, raise error; if not full digit str, raise false
						flag_idstr = reply_userID_str.isdigit()
					except ValueError:
						pass
					except KeyError:
						pass
					except AttributeError:
						pass
					except TypeError:
						pass
					else:
						if flag_idstr == True:
							Tweet_OBJ['reply_to_userID'].add(reply_userID_str)
					# if EE failed, set default value
					if len(Tweet_OBJ['reply_to_userID']) == 0:
						Tweet_OBJ['reply_to_userID'].add("0")

	#############################################################
	# extract tags from entities
	flag_tagExist = False # flag for tags of tweet
	if flag_TidTimeAuthor:
		# extract tags from entities
		tag_list = set([]) # eliminate repeating tags
		try:
			entities_json = tweet_json['entities']
			Hashtags_json = entities_json['hashtags']
			flag_tagExist = True
		except ValueError:
			pass
		except KeyError:
			pass
		except TypeError:
			pass
		else:		
			for entry in Hashtags_json:
				try:
					# THIS IS VERY VERY VERY IMPORTANT !!!!!
					tag_text = str(entry['text']).lower()
					if len(tag_text) > 253:
						tag_text = tag_text[:250]
					tag_list.add(tag_text) # THIS IS VERY VERY VERY IMPORTANT !!!!!
					# THIS IS VERY VERY VERY IMPORTANT !!!!!
					# MySQL cant distinguish upper and lower cases when str is used as name for table
					# which will result in confusion in data analysis
				except ValueError:
					pass
				except KeyError:
					pass
				except TypeError:
					pass
			# check if there is anything in tag_list
			if len(tag_list) == 0:
				flag_tagExist = False

	#############################################################
	# check if has tag with keyword 'trump'
	flag_tag_keyword1 = False
	flag_tag_keyword2 = False
	
	if flag_TidTimeAuthor and flag_tagExist:
		# check keyword_tag
		for item in tag_list:
			# keyword1
			if keyword1 in item:
				print "\n\nLine: {}, Key Word Tag found: {}".format(index_line, item)
				print "{} Tracking tags: ".format(keyword1), len(RollingScoreBank['tag_relevant1'])
				print "{} Tracking users: ".format(keyword1), len(RollingScoreBank['user1'])
				flag_tag_keyword1 = True
				# add to relevent tag set
				Tweet_OBJ['Tag_Keyword'].add(item)
			# keyword2
			if keyword2 in item:
				print "\n\nLine: {}, Key Word Tag found: {}".format(index_line, item)
				print "{} Tracking tags: ".format(keyword2), len(RollingScoreBank['tag_relevant2'])
				print "{} Tracking users: ".format(keyword2), len(RollingScoreBank['user2'])
				flag_tag_keyword2 = True
				# add to relevent tag set
				Tweet_OBJ['Tag_Keyword'].add(item)
		# handling the rest tags
		if flag_tag_keyword1 or flag_tag_keyword2:
			# remove this tag from tag_list
			for item in Tweet_OBJ['Tag_Keyword']:
				tag_list.discard(item)
			# all else added to relevent set()
			for item in tag_list: 
				Tweet_OBJ['Tag_Relevant'].add(item)

	#############################################################
	# check if has tag in the bank of "related" tags
	flag_tag_relevant1 = False
	flag_tag_relevant2 = False	
	# has time and user_infor, has tag, but not keyword tag
	if flag_TidTimeAuthor and flag_tagExist and flag_tag_keyword1 and flag_tag_keyword2:
		# check if there is high-tier tag of keyword1
		for item in tag_list:
			if item in RollingScoreBank['tag_relevant1'] and RollingScoreBank['tag_relevant1'][item] >= thre_tag_mid:
				flag_tag_relevant1 = True
				break
		# check if there is high-tier tag of keyword2
		for item in tag_list:
			if item in RollingScoreBank['tag_relevant2'] and RollingScoreBank['tag_relevant2'][item] >= thre_tag_mid:
				flag_tag_relevant2 = True
				break
		# load related tags if yes
		if flag_tag_relevant1 or flag_tag_relevant2:
			for item in tag_list:
				Tweet_OBJ['Tag_Relevant'].add(item)

	#############################################################
	# check if this tweet is issued by recorded user
	flag_due_user = False
	# has tag, but not keyword or related tag, check if is recorded user
	if flag_TidTimeAuthor and flag_tagExist and len(Tweet_OBJ['Tag_Keyword']) == 0 and len(Tweet_OBJ['Tag_Relevant']) == 0:
		if user_id in RollingScoreBank['user1'] and RollingScoreBank['user1'][user_id] >= thre_user_mid:
			flag_due_user = True
		if user_id in RollingScoreBank['user2'] and RollingScoreBank['user2'][user_id] >= thre_user_mid:
			flag_due_user = True
		# load ALL tags as due_to_user
		for item in tag_list: 
			Tweet_OBJ['Tag_due_user'].add(item)

	#############################################################
	# summarize flags
	flag_TweetStack = False
	if flag_TidTimeAuthor and flag_tagExist:
		if flag_tag_keyword1 or flag_tag_keyword2 or flag_tag_relevant1 or flag_tag_relevant2 or flag_due_user:
			flag_TweetStack = True

	#############################################################
	# extract text
	if flag_TweetStack:
		# extract date-time from mainbody
		try:
			text_str = tweet_json['text']
			text_str = transASC(text_str)
			text_str = removeUtf(text_str)
			text_str = text_str.replace("'", "")
			text_str = text_str.replace("#", "")
			text_str = parse_MultiLine_text(text_str)
		except ValueError:
			pass
		except KeyError:
			pass
		else:
			Tweet_OBJ['text'].add(text_str)

	#############################################################
	# extract mentioned_userID
	if flag_TweetStack:
		# extract entities and user_mentions	
		try:
			usermentions_json = entities_json['user_mentions']
		except ValueError:
			pass
		except KeyError:
			pass
		except TypeError:
			pass
		else:
			for entry in usermentions_json:
				try:
					Tweet_OBJ['mentioned_userID'].add(entry['id_str'])
				except ValueError:
					pass
				except KeyError:
					pass
				except TypeError:
					pass

	#############################################################
	return flag_TweetStack, Tweet_OBJ




"""
####################################################################
# test code for Stage1 main
"""



