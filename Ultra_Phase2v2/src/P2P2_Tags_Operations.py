

import json
import os
import numpy as np 
import statistics
import math
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
    # the same as Expanded_Tags = col.defaultdict(dict)
    MarkedTags_dict = col.defaultdict(dict)

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
            # initialize 
            if tag_text not in MarkedTags_dict:
                MarkedTags_dict[tag_text] = dict()
            # insert value
            # keyword 1
            if counter_keyword == 1:
                MarkedTags_dict[tag_text]['senti_score1'] = float(row[1])
                MarkedTags_dict[tag_text]['senti_score2'] = 0.0
            # keyword 2
            if counter_keyword == 2:
                MarkedTags_dict[tag_text]['senti_score1'] = 0.0
                MarkedTags_dict[tag_text]['senti_score2'] = float(row[1])
    # return dict()
    return MarkedTags_dict


'''
####################################################################
'''

def Get_Table_Names(MySQL_DBkey, pin_time, 
                    header='networkedtags_tagunique'):
    '''
    get table names
    For Phase2 Part2, this Get_Table_Names() should extract a single table name per day
    act as a check of existance

    header: 'networkedtags_tagunique'
    pin_time: the time pin for networkedtags_tagunique_
              pd.to_datetime(Start_Time)
    '''

    ####################################################################

    # Connect to the database
    connection = pymysql.connect(host=MySQL_DBkey['host'],
                                 user=MySQL_DBkey['user'],
                                 password=MySQL_DBkey['password'],
                                 db=MySQL_DBkey['db'],
                                 charset=MySQL_DBkey['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)
    
    ####################################################################

    db_name = MySQL_DBkey['db']

    # table to check
    table_name = header + pin_time.strftime('_%Y_%m_%d_%H')
    # comd
    comd_table_check = """
SELECT IF( 
(SELECT count(*) FROM information_schema.tables
WHERE table_schema = '%s' AND table_name = '%s'), 1, 0);"""
    # execute command
    try:
        with connection.cursor() as cursor:
            cursor.execute( comd_table_check % tuple( [db_name] + [table_name] )
                          )
            result = cursor.fetchall()
            #print result
            for key in result[0]:
                pin = result[0][key]
            #print pin
    finally:
        pass
    # load results
    if pin == 1:
        print "%s exists." % table_name
    else:
        print "%s does NOT exist." % table_name 
        table_name = None

    # end of loop
    connection.close()
    return table_name


'''
####################################################################
'''

def Expanded_TagScores_Extract(MySQL_DBkey, table_name):
    '''
    extract information from a given list of UserUnique_time tables
    extracted information: tagText, senti_score1, senti_score2, totalCall, 
                           score1_ave, score1_median, score2_ave, score2_median
    
    table_name: networkedtags_tagunique_2016_MM_dd_00
    '''
    # basic data structre
    # key as tagText
    Expanded_Tags = col.defaultdict(dict)

    ####################################################################

    # Connect to the database
    connection = pymysql.connect(host=MySQL_DBkey['host'],
                                 user=MySQL_DBkey['user'],
                                 password=MySQL_DBkey['password'],
                                 db=MySQL_DBkey['db'],
                                 charset=MySQL_DBkey['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)
    
    ####################################################################

    ################################################
    # extract data from relevent UserUnique tables #
    ################################################

    ####################################################################

    # Comd to extract data from each table
    Comd_Extract = """
SELECT tagText, senti_score1, senti_score2, totalCall, \
score1_ave, score1_median, score2_ave, score2_median
FROM %s"""
    # Execute Comds
    print "extracting tags from %s" % table_name
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_Extract % table_name )
            result = cursor.fetchall()
            # loop through all rows of this table
            for entry in result:
                # {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
                # in MySQL, tag format is utf8, but in raw data as ASCII
                tagText = str( entry['tagText'] ).decode('utf-8').encode('ascii', 'ignore') 

                # numericals
                Expanded_Tags[tagText]['senti_score1'] = entry['senti_score1']
                Expanded_Tags[tagText]['senti_score2'] = entry['senti_score2']
                Expanded_Tags[tagText]['totalCall'] = entry['totalCall']
                Expanded_Tags[tagText]['score1_ave'] = entry['score1_ave']
                Expanded_Tags[tagText]['score1_median'] = entry['score1_median']
                Expanded_Tags[tagText]['score2_ave'] = entry['score2_ave']
                Expanded_Tags[tagText]['score2_median'] = entry['score2_median']
    finally:
        pass

    ####################################################################
    print "%i tags extracted" % len(Expanded_Tags)
    connection.close()
    return Expanded_Tags


'''
####################################################################
'''

def tweetStack_Filtered_Extract(MySQL_DBkey, start_time, end_time, 
                                MarkedTags_dict, Expanded_Tags_dict, 
                                # variables on text filter
                                thre_nonTagWords=10, flag_ridTags=False,
                                save_tableName_header='ScoredFiltered_Tweets'):
    '''
    extract tweets from tweetStack between start_time and end_time
    apply filters on extracted tweets:  tags in either dicts 
                                        tweet text satisfy requirements
    calculate scores for tweets using various methods:
        HMS_NW: hand-marked-score, no-weight-assigned
        ES_NW: expanded-score, no-weight
        ES_CW: expanded-score, totalCall-weight
        ES_RW: expanded-score, relevence_score-weight
    save to data base

    --------
    start_time: the time pin for networkedtags_tagunique_
                pd.to_datetime(Start_Time)

    MarkedTags_dict: tagText, makred_senti_score1, marked_senti_score2
    Expanded_Tags_dict: tagText, senti_score1, senti_score2, totalCall, 
                        score1_ave, score1_median, score2_ave, score2_median

    thre_nonTagWords: threshold on counts of words that are not: user, https, 
    '''

    ####################################################################

    # Connect to the database
    connection = pymysql.connect(host=MySQL_DBkey['host'],
                                 user=MySQL_DBkey['user'],
                                 password=MySQL_DBkey['password'],
                                 db=MySQL_DBkey['db'],
                                 charset=MySQL_DBkey['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)
    
    ####################################################################
    
    ##################################
    # extract tweets from tweetStack #
    ##################################

    tweet_list_raw = []

    # convert time to string format
    start_time_str = start_time.strftime('%Y-%m-%d %H:%d:%m')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%d:%m')

    # Comd to extract data from each table
    Comd_tweetExtract = """
SELECT tweetID, tweetTime, userID, tweetText, taglist_text
FROM tweet_stack
WHERE tweetTime >= '%s' AND tweetTime < '%s';"""
    # Execute Comds
    print "extracting tweets during %s and %s" % tuple( [start_time_str] + [end_time_str] )
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_tweetExtract % tuple( [start_time_str] + [end_time_str] ) )
            result = cursor.fetchall()
            # loop through all rows of this table
            for entry in result:
                # {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
                tweetID_str = str( entry['tweetID'] )
                tweetTime_str = str( entry['tweetTime'] )
                userID_str = str( entry['userID'] )
                # in MySQL, tag format is utf8, but in raw data as ASCII
                tweetText = str( entry['tweetText'] ).decode('utf-8').encode('ascii', 'ignore') 
                taglist_str = str( entry['taglist_text'] ).decode('utf-8').encode('ascii', 'ignore')    
                # add into tweet_list_raw
                tweet_list_raw.append( [tweetID_str, tweetTime_str, userID_str, tweetText, taglist_str] )

    finally:
        pass
    print "%i tweets extracted" % len(tweet_list_raw)

    ####################################################################
    
    #########################################
    # filter out tweets without scored tags #
    #########################################

    tweet_list_postTagFilter = []
    # go through tweet_list_raw
    # [tweetID_str, tweetTime_str, userID_str, tweetText, taglist_str]
    for tweet in tweet_list_raw:
        taglist = tweet[4].split(',')
        flag_hasTag = False
        # through taglist
        for tag in taglist:
            if (tag in Expanded_Tags_dict) and (tag in MarkedTags_dict):
                flag_hasTag = True
        # append or not
        if flag_hasTag == True: 
            tweet_list_postTagFilter.append( tweet )
    
    print "%i tweets past Tags Filter by both Dicts" % len(tweet_list_postTagFilter)

    ####################################################################
    
    ##############################
    # apply filters on tweetText #
    ##############################

    tweet_list_postTextFilter = []
    # go through tweet_list_raw
    # [tweetID_str, tweetTime_str, userID_str, tweetText, taglist_str]
    for tweet in tweet_list_postTagFilter:
        taglist = tweet[4].split(',')
        tweetText_wordList = tweet[3].split(' ')
        tweetText_new = ""

        # get rid of tags of this tweet; more strict thre_nonTagWords
        if flag_ridTags == True:
            for tag in taglist:
                if (tag in MarkedTags_dict) or (tag in Expanded_Tags_dict):
                    tweetText_wordList.remove(tag)
        
        # filter on '@' and 'https'
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
        tweet[3] = tweetText_new        
        # check against thre_nonTagWords
        if len(tweetText_wordList) >= thre_nonTagWords: 
            tweet_list_postTextFilter.append( tweet )

    print "%i tweets post Tweet Text Filter" % len(tweet_list_postTextFilter)

    ####################################################################
    
    ############################################
    # convert tweet_list_postTextFilter        #
    # from list[list] to col.defaultdict(dict) #
    ############################################
    
    postFilter_tweet = col.defaultdict(dict)

    # [tweetID_str, tweetTime_str, userID_str, tweetText, taglist_str]
    for tweet in tweet_list_postTextFilter:
        postFilter_tweet[tweet[0]] = dict()
        postFilter_tweet[tweet[0]]['tweetTime'] = tweet[1]
        postFilter_tweet[tweet[0]]['userID'] = tweet[2]
        postFilter_tweet[tweet[0]]['tweetText'] = tweet[3]
        postFilter_tweet[tweet[0]]['taglist'] = tweet[4]

    ####################################################################
    
    ##########################################
    # calculate scores using MarkedTags_dict #
    ##########################################

    ####################################################################

    #############################################
    # calculate using hand-marked tags directly #
    # no weights assigned                       #
    #############################################

    for tweetID in postFilter_tweet:
        # initialize
        tweet_senti_score1 = 0.0
        tweet_senti_score2 = 0.0
        count_MarkedTags = 0
        # go through taglist
        taglist = postFilter_tweet[tweetID]['taglist'].split(',')
        for tag in taglist:
            if tag in MarkedTags_dict:
                count_MarkedTags += 1
                tweet_senti_score1 += MarkedTags_dict[tag]['senti_score1']
                tweet_senti_score2 += MarkedTags_dict[tag]['senti_score2']
        # add into tweet
        # if found tags, normalize using count_MarkedTags
        # HMS_NW: hand-marked-score, no-weight-assigned
        if count_MarkedTags > 0:
            postFilter_tweet[tweetID]['HMS_NW_senti_score1'] = 1.0*tweet_senti_score1/count_MarkedTags
            postFilter_tweet[tweetID]['HMS_NW_senti_score2'] = 1.0*tweet_senti_score2/count_MarkedTags  
        if count_MarkedTags == 0:
            postFilter_tweet[tweetID]['HMS_NW_senti_score1'] = 0.0
            postFilter_tweet[tweetID]['HMS_NW_senti_score2'] = 0.0              

    ##########################################
    # calculate using expanded scores        #
    # no weight                              #
    ##########################################
    
    for tweetID in postFilter_tweet:
        # initialize
        list_score1 = []
        list_score2 = []
        # go through taglist
        taglist = postFilter_tweet[tweetID]['taglist'].split(',')
        for tag in taglist:
            # Expanded_Tags_dict is floating hand-marked values
            if tag in Expanded_Tags_dict:
                # scores
                list_score1.append( Expanded_Tags_dict[tag]['senti_score1'] )
                list_score2.append( Expanded_Tags_dict[tag]['senti_score2'] )
            if (tag not in Expanded_Tags_dict) and (tag in MarkedTags_dict):
                list_score1.append( MarkedTags_dict[tag]['senti_score1'] )
                list_score2.append( MarkedTags_dict[tag]['senti_score2'] )              
            # end of if tag in Expanded_Tags_dict
        # end of for tag in taglist
        # if tag scores found
        # ES_NW: expanded-score, no-weight
        if len(list_score1) > 1:
            postFilter_tweet[tweetID]['ES_NW_senti_score1'] = statistics.mean(list_score1)
            postFilter_tweet[tweetID]['ES_NW_senti_score2'] = statistics.mean(list_score2)
            postFilter_tweet[tweetID]['ES_NW_senti_EoM1'] = statistics.stdev(list_score1)/math.sqrt(len(list_score1))
            postFilter_tweet[tweetID]['ES_NW_senti_EoM2'] = statistics.stdev(list_score2)/math.sqrt(len(list_score2))
        if len(list_score1) == 1:
            postFilter_tweet[tweetID]['ES_NW_senti_score1'] = statistics.mean(list_score1)
            postFilter_tweet[tweetID]['ES_NW_senti_score2'] = statistics.mean(list_score2)
            postFilter_tweet[tweetID]['ES_NW_senti_EoM1'] = 0.0
            postFilter_tweet[tweetID]['ES_NW_senti_EoM2'] = 0.0
        if len(list_score1) == 0:
            postFilter_tweet[tweetID]['ES_NW_senti_score1'] = 0.0
            postFilter_tweet[tweetID]['ES_NW_senti_score2'] = 0.0
            postFilter_tweet[tweetID]['ES_NW_senti_EoM1'] = 0.0
            postFilter_tweet[tweetID]['ES_NW_senti_EoM2'] = 0.0

    ##########################################
    # calculate using expanded scores        #
    # weight as per-day totalCall percentage #
    ##########################################

    for tweetID in postFilter_tweet:
        # count on scored-tags' cumulated totalCall
        count_totalCall_cumulate = 0
        list_score1 = []
        list_score2 = []
        # go through taglist
        taglist = postFilter_tweet[tweetID]['taglist'].split(',')
        for tag in taglist:
            # Expanded_Tags_dict is floating hand-marked values
            if tag in Expanded_Tags_dict:
                # cumulated totalCall
                count_totalCall_cumulate += Expanded_Tags_dict[tag]['totalCall']
                # scores
                list_score1.append( Expanded_Tags_dict[tag]['senti_score1']*Expanded_Tags_dict[tag]['totalCall'] )
                list_score2.append( Expanded_Tags_dict[tag]['senti_score2']*Expanded_Tags_dict[tag]['totalCall'] )
            # end of if tag in Expanded_Tags_dict
        # end of for tag in taglist
        # if tag scores found
        # ES_CW: expanded-score, totalCall-weight
        if len(list_score1) > 1:
            list_score1 = [ 1.0*value/count_totalCall_cumulate for value in list_score1]
            list_score2 = [ 1.0*value/count_totalCall_cumulate for value in list_score2]
            postFilter_tweet[tweetID]['ES_CW_senti_score1'] = statistics.mean(list_score1)
            postFilter_tweet[tweetID]['ES_CW_senti_score2'] = statistics.mean(list_score2)
            postFilter_tweet[tweetID]['ES_CW_senti_EoM1'] = statistics.stdev(list_score1)/math.sqrt( len(list_score1) )
            postFilter_tweet[tweetID]['ES_CW_senti_EoM2'] = statistics.stdev(list_score2)/math.sqrt( len(list_score2) )
        if len(list_score1) == 1:
            list_score1 = [ 1.0*value/count_totalCall_cumulate for value in list_score1]
            list_score2 = [ 1.0*value/count_totalCall_cumulate for value in list_score2]
            postFilter_tweet[tweetID]['ES_CW_senti_score1'] = statistics.mean(list_score1)
            postFilter_tweet[tweetID]['ES_CW_senti_score2'] = statistics.mean(list_score2)
            postFilter_tweet[tweetID]['ES_CW_senti_EoM1'] = 0.0
            postFilter_tweet[tweetID]['ES_CW_senti_EoM2'] = 0.0
        if len(list_score1) == 0:
            postFilter_tweet[tweetID]['ES_CW_senti_score1'] = 0.0
            postFilter_tweet[tweetID]['ES_CW_senti_score2'] = 0.0
            postFilter_tweet[tweetID]['ES_CW_senti_EoM1'] = 0.0
            postFilter_tweet[tweetID]['ES_CW_senti_EoM2'] = 0.0

    ###################################
    # calculate using expanded scores #
    # weight as relevence percentage  #
    ###################################
    
    for tweetID in postFilter_tweet:
        # cumulated percentage scores
        cumulate_percentage1 = 0.0
        cumulate_percentage2 = 0.0
        list_score1 = []
        list_score2 = []
        # go through taglist
        taglist = postFilter_tweet[tweetID]['taglist'].split(',')
        for tag in taglist:
            # Expanded_Tags_dict is floating hand-marked values
            if tag in Expanded_Tags_dict:
                shifted_Relev_score1 = (Expanded_Tags_dict[tag]['score1_ave'] + Expanded_Tags_dict[tag]['score1_median'])/2.0
                shifted_Relev_score2 = (Expanded_Tags_dict[tag]['score2_ave'] + Expanded_Tags_dict[tag]['score2_median'])/2.0
                # cumulated relevence score
                cumulate_percentage1 += shifted_Relev_score1
                cumulate_percentage2 += shifted_Relev_score2
                list_score1.append(Expanded_Tags_dict[tag]['senti_score1']*shifted_Relev_score1)
                list_score2.append(Expanded_Tags_dict[tag]['senti_score2']*shifted_Relev_score2)
            # end of if tag in Expanded_Tags_dict
        # end of for tag in taglist
        # if tag scores found
        # ES_RW: expanded-score, relevence_score-weight
        if len(list_score1) > 1:    
            list_score1 = [ 1.0*value/cumulate_percentage1 for value in list_score1]
            list_score2 = [ 1.0*value/cumulate_percentage2 for value in list_score2]
            postFilter_tweet[tweetID]['ES_RW_senti_score1'] = statistics.mean(list_score1)
            postFilter_tweet[tweetID]['ES_RW_senti_score2'] = statistics.mean(list_score2)
            postFilter_tweet[tweetID]['ES_RW_senti_EoM1'] = statistics.stdev(list_score1)/math.sqrt( len(list_score1) )
            postFilter_tweet[tweetID]['ES_RW_senti_EoM2'] = statistics.stdev(list_score2)/math.sqrt( len(list_score2) )
        if len(list_score1) == 1:   
            list_score1 = [ 1.0*value/cumulate_percentage1 for value in list_score1]
            list_score2 = [ 1.0*value/cumulate_percentage2 for value in list_score2]
            postFilter_tweet[tweetID]['ES_RW_senti_score1'] = statistics.mean(list_score1)
            postFilter_tweet[tweetID]['ES_RW_senti_score2'] = statistics.mean(list_score2)
            postFilter_tweet[tweetID]['ES_RW_senti_EoM1'] = 0.0
            postFilter_tweet[tweetID]['ES_RW_senti_EoM2'] = 0.0 
        if len(list_score1) == 0:
            postFilter_tweet[tweetID]['ES_RW_senti_score1'] = 0.0
            postFilter_tweet[tweetID]['ES_RW_senti_score2'] = 0.0
            postFilter_tweet[tweetID]['ES_RW_senti_EoM1'] = 0.0
            postFilter_tweet[tweetID]['ES_RW_senti_EoM2'] = 0.0

    ####################################################################

    #####################
    # save to data base #
    #####################

    # create New table
    # table Name
    tableName = save_tableName_header + start_time.strftime('_%Y_%m_%d_%H')
    #Comd
    Comd_ScoredFiltered_Tweets = """
DROP TABLE IF EXISTS %s;
CREATE TABLE IF NOT EXISTS %s
(
    tweetID BIGINT PRIMARY KEY NOT NULL,
    tweetTime TIMESTAMP,
    userID BIGINT,
    tweetText varchar(3000),
    taglist varchar(3000),
    HMS_NW_senti_score1 float, 
    HMS_NW_senti_score2 float,
    ES_NW_senti_score1 float, 
    ES_NW_senti_score2 float, 
    ES_NW_senti_EoM1 float, 
    ES_NW_senti_EoM2 float, 
    ES_CW_senti_score1 float, 
    ES_CW_senti_score2 float,
    ES_CW_senti_EoM1 float, 
    ES_CW_senti_EoM2 float,
    ES_RW_senti_score1 float, 
    ES_RW_senti_score2 float, 
    ES_RW_senti_EoM1 float, 
    ES_RW_senti_EoM2 float
)ENGINE=MEMORY;"""
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_ScoredFiltered_Tweets % tuple ( [tableName]+[tableName] )
                          )
        # commit commands
        print tableName + " Initialized"
        connection.commit()
    finally:
        pass

    # go through postFilter_tweet
    print "start loading %s" % tableName    
    for tweetID in postFilter_tweet:
        # command for Tweet_Stack
        comd_ScoredTweets_Insert = """
INSERT INTO %s (tweetID, tweetTime, userID, tweetText, taglist, \
HMS_NW_senti_score1, HMS_NW_senti_score2, \
ES_NW_senti_score1, ES_NW_senti_score2, ES_NW_senti_EoM1, ES_NW_senti_EoM2, \
ES_CW_senti_score1, ES_CW_senti_score2, ES_CW_senti_EoM1, ES_CW_senti_EoM2, \
ES_RW_senti_score1, ES_RW_senti_score2, ES_RW_senti_EoM1, ES_RW_senti_EoM2)
VALUES (%s, '%s', %s, '%s', '%s', \
%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, \
%.3f, %.3f, %.3f, %.3f, \
%.3f, %.3f, %.3f, %.3f)
ON DUPLICATE KEY UPDATE userID = %s;"""
        # execute commands
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_ScoredTweets_Insert % tuple( [tableName] + 
                                                                  [tweetID] +
                                                                  [postFilter_tweet[tweetID]['tweetTime']] + 
                                                                  [postFilter_tweet[tweetID]['userID']] + 
                                                                  [postFilter_tweet[tweetID]['tweetText']] + 
                                                                  [postFilter_tweet[tweetID]['taglist']] + 
                                                                  [postFilter_tweet[tweetID]['HMS_NW_senti_score1']] + 
                                                                  [postFilter_tweet[tweetID]['HMS_NW_senti_score2']] + 
                                                                  [postFilter_tweet[tweetID]['ES_NW_senti_score1']] + 
                                                                  [postFilter_tweet[tweetID]['ES_NW_senti_score2']] + 
                                                                  [postFilter_tweet[tweetID]['ES_NW_senti_EoM1']] + 
                                                                  [postFilter_tweet[tweetID]['ES_NW_senti_EoM2']] + 
                                                                  [postFilter_tweet[tweetID]['ES_CW_senti_score1']] + 
                                                                  [postFilter_tweet[tweetID]['ES_CW_senti_score2']] + 
                                                                  [postFilter_tweet[tweetID]['ES_CW_senti_EoM1']] + 
                                                                  [postFilter_tweet[tweetID]['ES_CW_senti_EoM2']] + 
                                                                  [postFilter_tweet[tweetID]['ES_RW_senti_score1']] + 
                                                                  [postFilter_tweet[tweetID]['ES_RW_senti_score2']] + 
                                                                  [postFilter_tweet[tweetID]['ES_RW_senti_EoM1']] + 
                                                                  [postFilter_tweet[tweetID]['ES_RW_senti_EoM2']] + 
                                                                  [postFilter_tweet[tweetID]['userID']]
                                                                ) 
                              )
            # commit commands
            connection.commit()
        finally:
            pass

    # convert to InnoDB
    #Comd
    comd_convert = """
ALTER TABLE %s ENGINE=InnoDB;"""
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( comd_convert % tableName )
        # commit commands
        print tableName + " Converted"
        connection.commit()
    finally:
        pass

    ####################################################################
    connection.close()
    return None


'''
####################################################################
'''

def ScoredTweets_Extract_2Way(MySQL_DBkey, table_name):
    '''
    tweets will be separated into: support trump, support hillar, cannot decide
    only support trump/hillary will be passed on
    
    HMS_NW_senti_scores and ES_NW_senti_scores are used
    '''
    
    ####################################################################

    # Connect to the database
    connection = pymysql.connect(host=MySQL_DBkey['host'],
                                 user=MySQL_DBkey['user'],
                                 password=MySQL_DBkey['password'],
                                 db=MySQL_DBkey['db'],
                                 charset=MySQL_DBkey['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)
    
    ####################################################################

    ##################
    # extract tweets # 
    ##################

    ExtractTweets_list = []

    # Comd to extract data from each table
    Comd_Extract = """
SELECT tweetID, tweetTime, tweetText, \
HMS_NW_senti_score1, HMS_NW_senti_score2, ES_NW_senti_score1, ES_NW_senti_score2
FROM %s;"""
    # Execute Comds
    print "extracting tweets from %s" % table_name
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_Extract % table_name )
            result = cursor.fetchall()
            # loop through all rows of this table
            for entry in result:
                # {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
                # in MySQL, tag format is utf8, but in raw data as ASCII
                tweetID_str = str( entry['tweetID'] ) 
                tweetTime_str = str( entry['tweetTime'] )
                tweetText = str( entry['tweetText'] ).decode('utf-8').encode('ascii', 'ignore') 
                tweetText = tweetText.lower()
                # numericals
                HMS_NW_senti_score1 = entry['HMS_NW_senti_score1']
                HMS_NW_senti_score2 = entry['HMS_NW_senti_score2']
                ES_NW_senti_score1 = entry['ES_NW_senti_score1']
                ES_NW_senti_score2 = entry['ES_NW_senti_score2']
                # add to ExtractTweets_list
                ExtractTweets_list.append( [tweetID_str, tweetTime_str, tweetText, \
                                            HMS_NW_senti_score1, HMS_NW_senti_score2, ES_NW_senti_score1, ES_NW_senti_score2] 
                                         )
    finally:
        pass

    print "%i tweets extracted" % len(ExtractTweets_list)

    ####################################################################

    ##################################
    # 2Way split + 1Way can't decide #
    ##################################

    SplitTweets_2Way = []

    for tweet in ExtractTweets_list:
        # initialize
        senti_flag = -1 # 1 for trump, 0 for hillary
        # [tweetID_str, tweetTime_str, tweetText, HMS_NW_senti_score1, HMS_NW_senti_score2, ES_NW_senti_score1, ES_NW_senti_score2]
        # check for trump
        if (tweet[3] > tweet[4]) and (tweet[5] > tweet[6]):
            senti_flag = 1
        # check for hillary
        if (tweet[3] < tweet[4]) and (tweet[5] < tweet[6]):
            senti_flag = 0  
        # add to SplitTweets_2Way
        # tweetID_str, tweetTime_str, tweetText, senti_flag
        if senti_flag != -1:
            SplitTweets_2Way.append( [tweet[0], tweet[1], tweet[2], senti_flag] )

    print "%i tweets splited 2Ways" % len(SplitTweets_2Way)

    ####################################################################
    connection.close()
    return SplitTweets_2Way 


'''
####################################################################
'''

def ScoredTweets_Load(MySQL_DBkey, table_name, tweet_list):
    '''
    '''
    
    ####################################################################

    # Connect to the database
    connection = pymysql.connect(host=MySQL_DBkey['host'],
                                 user=MySQL_DBkey['user'],
                                 password=MySQL_DBkey['password'],
                                 db=MySQL_DBkey['db'],
                                 charset=MySQL_DBkey['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)
    
    ####################################################################

    # create New table
    #Comd
    Comd_Prediction_Table = """
DROP TABLE IF EXISTS %s;
CREATE TABLE IF NOT EXISTS %s
(
    tweetID BIGINT PRIMARY KEY NOT NULL,
    tweetTime TIMESTAMP,
    tweetText varchar(3000),
    senti_flag int
)ENGINE=MEMORY;"""
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_Prediction_Table % tuple ( [table_name]+[table_name] )
                          )
        # commit commands
        print table_name + " Initialized"
        connection.commit()
    finally:
        pass

    # go through tweet_list
    # [tweetID_str, tweetTime_str, tweetText, senti_flag]
    print "start loading %s" % table_name   
    for tweet in tweet_list:
        # command for Tweet_Stack
        comd_ScoredTweets_Insert = """
INSERT INTO %s (tweetID, tweetTime, tweetText, senti_flag)
VALUES (%s, '%s', '%s', %i)
ON DUPLICATE KEY UPDATE tweetTime = '%s';"""
        # execute commands
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_ScoredTweets_Insert % tuple( [table_name] + 
                                                                  [tweet[0]] + 
                                                                  [tweet[1]] + 
                                                                  [tweet[2]] + 
                                                                  [tweet[3]] + 
                                                                  [tweet[1]]
                                                                ) 
                              )
            # commit commands
            connection.commit()
        finally:
            pass

    # convert to InnoDB
    #Comd
    comd_convert = """
ALTER TABLE %s ENGINE=InnoDB;"""
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( comd_convert % table_name )
        # commit commands
        print table_name + " Converted"
        connection.commit()
    finally:
        pass
    
    ####################################################################
    connection.close()
    return None 


'''
####################################################################
'''

def Load_Predictions(MySQL_DBkey, pred_columnName, sql_tableName, 
                     fileName_Scores_tuple, predictions_tuple):
    '''
    for loading predicted results from LSTM into data base

    pred_columnName: column name for prediction results
    sql_tableName: table name

    fileName_Scores_tuple: ( fileNames, scores_tag ), tuple of list
    predictions_tuple: ( dataset_preds_prob, dataset_preds), tuple of list

    '''

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
    print len_list
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












