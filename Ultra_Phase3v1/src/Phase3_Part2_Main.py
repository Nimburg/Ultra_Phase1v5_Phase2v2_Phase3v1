

import json
import os
import numpy as np 
import pandas as pd
import collections as col
from copy import deepcopy

import pymysql.cursors

from P3P2_MySQL_Operations import Get_Table_Names, PopularTags_Dictionary, DataPrep_perDayTweets
from P3P2_DataOpeartion import MarkedTag_Import


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
########################################################################################

Set up SQL in-RAM table variables

########################################################################################
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
########################################################################################

Phase3 Part2 Extract Hash Tags' Data and Prep for Learning

########################################################################################
"""

def Phase3_Part2_NetworkedTags(MySQL_DBkey, 
                               start_time, end_time, 
                               Tag_Extract_header='NetworkedTags_TagUnique',
                               Tweet_Extract_header='ScoredFiltered_Tweets',
                               Tag_Reload_header='SentiEstimate_TagUnique', # per day basis
                               size_perDayRanking=300,
                               size_tagDictionary=300,
                               flag_trainOrpredict=True, flag_forcePredict=True,
                               train_test_ratio=0.8,
                               DataSet_header='perDayTweets'):
    '''
    This function build dictionary of hash tags in descending order of usage for entire period
    Extract and build training dataset for linear/nonlinear methods on daily basis
    Create reloaded MySQL table of tags for comparison on daily basis
    Pickle final results

    start_time: Start_Time = pd.to_datetime(Start_Time)
    end_time: end of entire period
    time_period: in seconds, per Day
    
    size_perDayRanking: threshold for per-day tag usage
    dictionary_size: How many hash tags to estimate
    '''
    assert size_perDayRanking is not None
    assert size_tagDictionary is not None

    ####################################################################

    #################################
    # get NetworkedTags table names #
    #################################

    print "\n\nExtracting between %s and %s " % tuple( [start_time.strftime('_%Y_%m_%d_%H')] + 
                                                       [end_time.strftime('_%Y_%m_%d_%H')] 
                                                     )
    # extract table names from data base
    TableNames_AllTags = Get_Table_Names(MySQL_DBkey=MySQL_DBkey, 
                                         start_time=start_time, end_time=end_time, 
                                         header=Tag_Extract_header)

    #########################################
    # extract tags from each day by ranking #
    # accumulate and create ranked dict     #
    #########################################

    # tag2index as dict[tag] = (index, usage)
    # index2tag as list[(tag, usage)]  
    index2tag, tag2index = PopularTags_Dictionary(MySQL_DBkey=MySQL_DBkey, list_tableNames=TableNames_AllTags, 
                                                  size_perDayRanking=size_perDayRanking)

    ####################################################################

    #########################################
    # get ScoredFiltered_Tweets table names #
    #########################################

    print "\n\nExtracting between %s and %s " % tuple( [start_time.strftime('_%Y_%m_%d_%H')] + 
                                                       [end_time.strftime('_%Y_%m_%d_%H')] 
                                                     )
    TableNames_perDayTweets = Get_Table_Names(MySQL_DBkey=MySQL_DBkey, 
                                              start_time=start_time, end_time=end_time, 
                                              header=Tweet_Extract_header)

    #######################################
    # Extract perDay Tweets to create Y,X #
    #######################################

    for tableName in TableNames_perDayTweets:
        print "Extract, Prep and Pickle Tweets from %s" % tableName
        DataPrep_perDayTweets(MySQL_DBkey=MySQL_DBkey, tableName=tableName, 
                              index2tag=index2tag, tag2index=tag2index,
                              train_test_ratio=train_test_ratio,
                              size_tagDictionary=size_tagDictionary,
                              flag_trainOrpredict=flag_trainOrpredict, 
                              flag_forcePredict=flag_forcePredict,
                              DataSet_header=DataSet_header)

    ####################################################################

    return None


"""
########################################################################################

Phase3 Part2 Execution

########################################################################################
"""

if __name__ == "__main__":

    ####################################################################

    MySQL_DBkey = {'host':'localhost', 'user':'', 'password':'', 'db':'','charset':'utf8'}

    ####################################################################

    HandMarkedTags_fileNames_list = ['MarkedTag_keyword1.csv','MarkedTag_keyword2.csv']
    
    ####################################################################

    flag_DataPrep = True

    ####################################################################

    if flag_DataPrep == True: 

        Start_Time = '2016-10-31 00:00:00'
        # Start_Time = '2016-10-14 00:00:00'
        Start_Time = pd.to_datetime(Start_Time)
        print "start time: ", Start_Time.strftime('_%Y_%m_%d_%H')

        End_Time = '2016-11-11 00:00:00'
        # End_Time = '2016-11-30 00:00:00'
        End_Time = pd.to_datetime(End_Time)
        print "end time: ", End_Time.strftime('_%Y_%m_%d_%H')
        
        # go to next time point
        # pin_time = pin_time + np.timedelta64(3600,'s')
        Time_Period = np.timedelta64(3600*24,'s') # one day

        ####################################################################
        
        Phase3_Part2_NetworkedTags(MySQL_DBkey=MySQL_DBkey, 
                                   start_time=Start_Time, 
                                   end_time=End_Time, 
                                   Tag_Extract_header='NetworkedTags_TagUnique',
                                   Tweet_Extract_header='ScoredFiltered_Tweets',
                                   Tag_Reload_header='SentiEstimate_TagUnique', # per day basis
                                   size_perDayRanking=300,
                                   size_tagDictionary=300,
                                   flag_trainOrpredict=True, flag_forcePredict=True,
                                   train_test_ratio=0.8,
                                   DataSet_header='perDayTweets')

    ####################################################################






