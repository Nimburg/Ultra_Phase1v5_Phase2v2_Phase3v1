

import json
import os
import numpy as np 
import pandas as pd
import collections as col

from copy import deepcopy

import pymysql.cursors

from P3P3_MySQL_Operations import Get_Table_Names
from P3P3_MySQL_Operations import NetworkedTags_TagUnique_Concatenate, NetworkedTags_TagUnique_postIterSave
from P3P3_MySQL_Operations import NetworkedTags_Extract_RequestedTags

from P3P3_NetworkedTags import MarkedTag_Import, Iteration_TagSentiScores


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

TagUnique Functions of Phase3 Part1

####################################################################
"""

def Phase3_Part3_NetworkedTags(MySQL_DBkey, 
                               MarkedTags_fileName_list,
                               start_time, end_time, time_period, 
                               concatenated_tableName_header='NetworkedTags_TagUnique',
                               header='TagUnqiue'):
    '''
    this function concatenates those per-4 hour TagUnique tables into set-time-period TagUnique tables
    Extract statistic results from concatenated TagUnique tables
    output these results in .csv format

    start_time: Start_Time = pd.to_datetime(Start_Time)
    end_time: 
    time_period: in seconds
    header: for TagUnique_Concatenate_Extract(), header's default value is 'tagunique'
            but Get_Table_Names() could use other headers
    '''

    ###############################
    # import the hand-marked tags #
    ###############################

    MarkedTags_dict = MarkedTag_Import(file_name_list=MarkedTags_fileName_list)

    ####################################################################
    # list of post-concatenation table names
    Concatenated_TableNames = []
    # initialize for the first period
    pin_start_time = start_time
    pin_end_time = start_time + time_period
    # tagunique_'time'
    concatenated_tableName = concatenated_tableName_header + pin_start_time.strftime('_%Y_%m_%d_%H')

    # initialize col.defaultdict(col.defaultdict) for the 1st day
    Tags_Today_preIter = col.defaultdict(col.defaultdict)
    # post-Iteration col.defaultdict(col.defaultdict) of previous day
    Tags_PreviousDay_postIter = col.defaultdict(col.defaultdict)

    # while loop until pin_start_time is later than end_time
    while pin_start_time < end_time:

        #############################
        # get tagunique table names #
        #############################

        print "\n\nExtracting between %s and %s " % tuple( [pin_start_time.strftime('_%Y_%m_%d_%H')] + 
                                                               [pin_end_time.strftime('_%Y_%m_%d_%H')] 
                                                             )
        print "size of Tags_PreviousDay_postIter: %i \n\n" % len(Tags_PreviousDay_postIter)

        # extract table names from data base
        list_table_names = Get_Table_Names(MySQL_DBkey=MySQL_DBkey, 
                                           start_time=pin_start_time, end_time=pin_end_time, 
                                           header=header)
        # if tables are not found
        if len(list_table_names) == 0:
            print "No TagUnique_ talbes found between %s and %s" % tuple( [pin_start_time.strftime('_%Y_%m_%d_%H')] + 
                                                                          [pin_end_time.strftime('_%Y_%m_%d_%H')] 
                                                                        )

        #########################################################################
        # extract tags within time period into col.defaultdict(col.defaultdict) #
        #########################################################################
        
        # if tables are found
        if len(list_table_names) > 0:
            # create concatenated tagunique_'' table for this period
            # return defaultdict_All_Tags for this period
            Tags_Today_preIter = NetworkedTags_TagUnique_Concatenate(MySQL_DBkey=MySQL_DBkey, 
                                                                     list_tableNames=list_table_names)

            print "\n number of tags inside Tags_Today_preIter: %i \n" % len(Tags_Today_preIter)

            #########################################################
            # calculate Senti_Scores for tags in Tags_Today_preIter #
            # through Iteration; then load into Tags_Today_postIter #
            #########################################################

            Tags_Today_postIter = Iteration_TagSentiScores(MarkedTags_dict=MarkedTags_dict, 
                                                           Tags_Today_preIter=Tags_Today_preIter, 
                                                           Tags_PreviousDay_postIter=Tags_PreviousDay_postIter)

            ########################################################
            # load Tags_Today_postIter into concatenated_tableName # 
            ########################################################

            NetworkedTags_TagUnique_postIterSave(MySQL_DBkey=MySQL_DBkey, 
                                                 New_tableName=concatenated_tableName,
                                                 Tags_Today_postIter=Tags_Today_postIter)

        ####################################################################
        # if tables are found
        if len(list_table_names) > 0:
            # only adding concatenated_tableName to Concatenated_TableNames
            # if tables are found
            Concatenated_TableNames.append(concatenated_tableName)
            # update Tags_PreviousDay_postIter
            Tags_PreviousDay_postIter = deepcopy( Tags_Today_postIter )

        # update pin_start_time
        pin_start_time = pin_start_time + time_period
        pin_end_time = pin_end_time + time_period
        # update concatenated_tableName
        concatenated_tableName = concatenated_tableName_header + pin_start_time.strftime('_%Y_%m_%d_%H')

    # end of while pin_start_time < end_time
    ####################################################################

    ####################################################################
    return Concatenated_TableNames


"""
####################################################################

Phase3 Part3: extract required tags into .csv

####################################################################
"""

def Phase3_Part3_Extract_RequestedTags(MySQL_DBkey, 
                                       Requested_Tags_list,
                                       start_time, end_time, 
                                       NetworkedTags_tableName='NetworkedTags_TagUnique'
                                       ):
    '''
    Requested_Tags_list: 
    '''

    # list of NetworkedTags_TagUnique_per-day table names
    list_TableNames = []

    print "\n\nExtracting between %s and %s \n\n" % tuple( [start_time.strftime('_%Y_%m_%d_%H')] + 
                                                           [end_time.strftime('_%Y_%m_%d_%H')] 
                                                         )
    # extract table names from data base
    list_TableNames = Get_Table_Names(MySQL_DBkey=MySQL_DBkey, 
                                       start_time=start_time, end_time=end_time, 
                                       header=NetworkedTags_tableName)

    ####################################################################
    # extract to .csv for each tag in Requested_Tags_list
    for tag in Requested_Tags_list:

        NetworkedTags_Extract_RequestedTags(MySQL_DBkey=MySQL_DBkey, 
                                            tag_text=tag, 
                                            postIter_taleName_list=list_TableNames)

    ####################################################################
    return None


"""
####################################################################

# Execution of Phase2_Part1

####################################################################
"""

if __name__ == "__main__":

    ####################################################################

    MySQL_DBkey = {'host':'localhost', 'user':'sa', 'password':'fanyu01', 'db':'ultrajuly_p1v5_p2v2','charset':'utf8'}

    ####################################################################

    flag_Generate_NetworkedTags = False  

    HandMarkedTags_fileNames_list = ['MarkedTag_keyword1.csv','MarkedTag_keyword2.csv']
    
    ####################################################################    

    flag_Extract_RequestedTags = True  

    Trump_list = ['trump', 'donaldtrump', 'trump2016', 'trumppence16', 'trumppence2016', 'trumppence', 'trumptrain']
    Hillary_list = ['hillary', 'hillaryclinton', 'hillary2016', 'clinton', 'demsinphilly']

    ####################################################################

    if flag_Generate_NetworkedTags == True: 

        Start_Time = '2016-07-12 00:00:00'
        # Start_Time = '2016-10-14 00:00:00'
        Start_Time = pd.to_datetime(Start_Time)
        print "start time: ", Start_Time.strftime('_%Y_%m_%d_%H')

        End_Time = '2016-11-05 00:00:00'
        # End_Time = '2016-11-30 00:00:00'
        End_Time = pd.to_datetime(End_Time)
        print "end time: ", End_Time.strftime('_%Y_%m_%d_%H')
        
        # go to next time point
        # pin_time = pin_time + np.timedelta64(3600,'s')
        Time_Period = np.timedelta64(3600*24,'s') # one day
        
        ####################################################################
        
        Phase3_Part3_NetworkedTags(MySQL_DBkey=MySQL_DBkey, 
                                   MarkedTags_fileName_list=HandMarkedTags_fileNames_list,
                                   start_time=Start_Time, end_time=End_Time, 
                                   time_period=Time_Period, 
                                   concatenated_tableName_header='NetworkedTags_TagUnique',
                                   header='tagunique'
                                   )

    ####################################################################

    if flag_Extract_RequestedTags == True: 

        Start_Time = '2016-07-12 00:00:00'
        # Start_Time = '2016-10-14 00:00:00'
        Start_Time = pd.to_datetime(Start_Time)
        print "start time: ", Start_Time.strftime('_%Y_%m_%d_%H')

        End_Time = '2016-11-05 00:00:00'
        # End_Time = '2016-11-30 00:00:00'
        End_Time = pd.to_datetime(End_Time)
        print "end time: ", End_Time.strftime('_%Y_%m_%d_%H')
        
        # go to next time point
        # pin_time = pin_time + np.timedelta64(3600,'s')
        Time_Period = np.timedelta64(3600*24,'s') # one day

        ####################################################################
        # trump
        Phase3_Part3_Extract_RequestedTags(MySQL_DBkey=MySQL_DBkey, 
                                           Requested_Tags_list=Trump_list,
                                           start_time=Start_Time, end_time=End_Time, 
                                           NetworkedTags_tableName='NetworkedTags_TagUnique' 
                                           )
        # hillary
        Phase3_Part3_Extract_RequestedTags(MySQL_DBkey=MySQL_DBkey, 
                                           Requested_Tags_list=Hillary_list,
                                           start_time=Start_Time, end_time=End_Time, 
                                           NetworkedTags_tableName='NetworkedTags_TagUnique' 
                                           )

    ####################################################################









