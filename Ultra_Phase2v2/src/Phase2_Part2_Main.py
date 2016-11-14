

import json
import sys
import time
import os
import numpy as np 
import pandas as pd
import collections as col
import csv
import itertools

import nltk
import pymysql.cursors

from P2P2_Tags_Operations import MarkedTag_Import
from P2P2_Tags_Operations import Get_Table_Names, Expanded_TagScores_Extract
from P2P2_Tags_Operations import tweetStack_Filtered_Extract
from P2P2_Tags_Operations import ScoredTweets_Extract_2Way, ScoredTweets_Load
from P2P2_Tokenize import NLTK_Tokenize_DataPickle


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

Main Function of Phase2_Part2

####################################################################
"""

def Pahse2_Part2_Main(MySQL_DBkey, file_name_list, 
                      start_time, end_time):
    '''
    file_name_list: file name lists .csv for hand-marked tags
    '''

    # set in-RAM table size
    Set_TempTable_Variables(MySQL_DBkey = MySQL_DBkey, N_GB = 4)

    ####################################################################

    #################################
    # get dict for hand-marked tags #
    #################################

    MarkedTags_dict = MarkedTag_Import(file_name_list=file_name_list)

    ####################################################################

    ##################################
    # going through each time period #
    ##################################

    pin_time = start_time
    # loop through time
    while (pin_time < end_time):

        # get full table name for this day
        table_name = Get_Table_Names(MySQL_DBkey=MySQL_DBkey, pin_time=pin_time, 
                                     header='networkedtags_tagunique')  
        # pass this pin_time is table not found
        if table_name is None:
            # go to next time point
            pin_time = pin_time + np.timedelta64(1,'D') # one day           
            continue
        # continue analysis if table is found
        if table_name is not None: 
            # get dict of expanded scores of tags
            Expanded_Tags = Expanded_TagScores_Extract(MySQL_DBkey=MySQL_DBkey, table_name=table_name)
            end_pin_time = pin_time + np.timedelta64(1,'D')
            tweetStack_Filtered_Extract(MySQL_DBkey=MySQL_DBkey, 
                                        start_time=pin_time, end_time=end_pin_time, 
                                        MarkedTags_dict=MarkedTags_dict, Expanded_Tags_dict=Expanded_Tags, 
                                        # variables on text filter
                                        thre_nonTagWords=10, flag_ridTags=False,
                                        save_tableName_header='ScoredFiltered_Tweets')
            # go to next time point
            pin_time = pin_time + np.timedelta64(1,'D') # one day
    # end of while (pin_time < end_time)

    return None


"""
####################################################################

Scored Tweets Extract and Tokenization

####################################################################
"""

def Phase2_Part2_LSTM_DataSet(MySQL_DBkey, start_time, end_time,
                              train_test_ratio, force_predict=False,
                              vocabulary_size=1000,
                              dataset_name='ScoredTweets'):
    '''
    Note: This function creates data sets for both training and predicting
    For predicting: ScoredFiltered_Tweets_ tables should already generated
    For predicting: vocabulary_size should inherit from its training LSTM parameters

    start_time: function will search between start/end time for ScoredFiltered_Tweets_
    end_time: 

    train_test_ratio: ratio of train/test
                      if train_test_ratio == 1, then all data are extracted for prediction
    force_predict: if True, then create prediction MySQL table even if train_test_ratio != 1
    '''

    # set in-RAM table size
    Set_TempTable_Variables(MySQL_DBkey = MySQL_DBkey, N_GB = 4)

    ####################################################################

    ##################################
    # extract scored tweets by time  #
    # going through each time period #
    ##################################

    TweetList_2Way = []

    pin_time = start_time
    # loop through time
    while (pin_time < end_time):
        # get full table name for this day
        table_name = Get_Table_Names(MySQL_DBkey=MySQL_DBkey, pin_time=pin_time, 
                                     header='scoredfiltered_tweets')    
        # pass this pin_time is table not found
        if table_name is None:
            # go to next time point
            pin_time = pin_time + np.timedelta64(1,'D') # one day           
            continue
        # continue analysis if table is found
        if table_name is not None: 
            # list of sentences
            New_Extract = ScoredTweets_Extract_2Way(MySQL_DBkey=MySQL_DBkey, table_name=table_name)
            # list of sentences
            TweetList_2Way = TweetList_2Way + New_Extract
            # go to next time point
            pin_time = pin_time + np.timedelta64(1,'D') # one day
    # end of while (pin_time < end_time)

    # train and test data set
    TweetList_train = []
    TweetList_test = []
    index_pin = train_test_ratio*len(TweetList_2Way)
    # shuffle data set  
    index_list = np.arange(len(TweetList_2Way))
    np.random.shuffle(index_list)
    # load train/test data set
    for idx in range(len(TweetList_2Way)):
        # training
        if idx <= index_pin:
            TweetList_train.append( TweetList_2Way[ index_list[idx] ]
                                  )
        # testing
        if idx > index_pin:
            TweetList_test.append( TweetList_2Way[ index_list[idx] ]
                                  )

    ####################################################################

    ####################################################
    # if for prediction, load back into MySQL database # 
    ####################################################

    flag_loadMySQL = False
    # by ratio or by force_predict
    if (train_test_ratio == 1) or (force_predict == True): 
        flag_loadMySQL = True

    tableName_predict = 'Prediction_' + dataset_name
    # load into MySQL tables for prediction
    if flag_loadMySQL == True:
        # load full list
        ScoredTweets_Load(MySQL_DBkey=MySQL_DBkey, table_name=tableName_predict, 
                          tweet_list=TweetList_2Way)

    ####################################################################

    ####################################
    # tokenize and pickle the data set #
    ####################################

    # check training or predicting
    flag_trainOrpredict = True
    if train_test_ratio == 1:
        flag_trainOrpredict = False
        TweetList_test = None

    # tokenize
    NLTK_Tokenize_DataPickle(Data_Train=TweetList_train, Data_Test=TweetList_test, Data_Full=TweetList_2Way, 
                             DataSet_Name=dataset_name, 
                             flag_trainOrpredict=flag_trainOrpredict, flag_forcePredict=force_predict,
                             vocabulary_size=vocabulary_size)

    ####################################################################
    return None


"""
####################################################################

# Execution of Phase2_Part2

####################################################################
"""

if __name__ == "__main__":

    ####################################################################

    MySQL_DBkey = {'host':'localhost', 'user':'', 'password':'', 'db':'','charset':'utf8'}
    
    ####################################################################

    HandMarkedTags_fileNames_list = ['MarkedTag_keyword1.csv','MarkedTag_keyword2.csv']
    
    Start_Time = '2016-07-12 00:00:00'
    # Start_Time = '2016-10-14 00:00:00'
    Start_Time = pd.to_datetime(Start_Time)
    print "start time: ", Start_Time.strftime('_%Y_%m_%d_%H')

    End_Time = '2016-11-10 00:00:00'
    # End_Time = '2016-11-30 00:00:00'
    End_Time = pd.to_datetime(End_Time)
    print "end time: ", End_Time.strftime('_%Y_%m_%d_%H')

    ####################################################################
    
    flag_ScoringTweets_perDay = False 

    flag_LSTM_DataPrep = True   

    ####################################################################

    if flag_ScoringTweets_perDay == True: 
        Pahse2_Part2_Main(MySQL_DBkey=MySQL_DBkey, 
                          file_name_list=HandMarkedTags_fileNames_list, 
                          start_time=Start_Time, end_time=End_Time)

    ####################################################################
    
    if flag_LSTM_DataPrep == True: 
        Phase2_Part2_LSTM_DataSet(MySQL_DBkey=MySQL_DBkey, start_time=Start_Time, end_time=End_Time,
                                  train_test_ratio=0.8, force_predict=True,
                                  vocabulary_size=60000,
                                  dataset_name='FullScoredTweets')

    ####################################################################
