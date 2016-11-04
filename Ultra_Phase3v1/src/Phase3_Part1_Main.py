

import json
import os
import numpy as np 
import pandas as pd
import collections as col

import pymysql.cursors

from P3P1_MySQL_Operations import Get_Table_Names, TagUnique_Concatenate, TagUnique_postConcatenation_Output

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

def TagUnique_Concatenate_Extract(MySQL_DBkey, 
                                  start_time, end_time, time_period, 
                                  csv_fileName,
                                  header='tagunique'):
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

    # list of post-concatenation table names
    Concatenated_TableNames = []
    # initialize for the first period
    pin_start_time = start_time
    pin_end_time = start_time + time_period
    # tagunique_'time'
    concatenated_tableName = 'concatenated_Tags' + pin_start_time.strftime('_%Y_%m_%d_%H')
    # as list holding col.defaultdict(dict)
    list_Period_TagDict = []

    # while loop until pin_start_time is later than end_time
    while pin_start_time < end_time:

        #############################
        # get tagunique table names #
        #############################

        print "Extracting between %s and %s" % tuple( [pin_start_time.strftime('_%Y_%m_%d_%H')] + 
                                                      [pin_end_time.strftime('_%Y_%m_%d_%H')] 
                                                    )
        # extract table names from data base
        list_table_names = Get_Table_Names(MySQL_DBkey=MySQL_DBkey, 
                                           start_time=pin_start_time, end_time=pin_end_time, 
                                           header=header)
        # if tables are not found
        if len(list_table_names) == 0:
            print "No TagUnique_ talbes found between %s and %s" % tuple( [pin_start_time.strftime('_%Y_%m_%d_%H')] + 
                                                                          [pin_end_time.strftime('_%Y_%m_%d_%H')] 
                                                                        )

        ##########################################
        # extract & save tags within time period #
        ##########################################
        
        # if tables are found
        if len(list_table_names) > 0:
            # create concatenated tagunique_'' table for this period
            # return defaultdict_All_Tags for this period
            TagUnique_Concatenate(MySQL_DBkey=MySQL_DBkey, 
                                  list_tableNames=list_table_names, 
                                  New_tableName=concatenated_tableName)

            # only adding concatenated_tableName to Concatenated_TableNames
            # if tables are found
            Concatenated_TableNames.append(concatenated_tableName)

            # extract to per-period .csv tables ???
            
        # update pin_start_time
        pin_start_time = pin_start_time + time_period
        pin_end_time = pin_end_time + time_period
        # update concatenated_tableName
        concatenated_tableName = 'concatenated_Tags' + pin_start_time.strftime('_%Y_%m_%d_%H')

    ####################################################################

    #######################################################
    # extract tags concatenated tables and output to .csv #
    #######################################################

    concatenated_period_relevence = []
    concatenated_period_hisCall = []

    # go through Concatenated_TableNames
    for table in Concatenated_TableNames: 
        print table
        # extract period_relevence, period_hisCall
        # as list of lists
        period_relevence, period_hisCall = TagUnique_postConcatenation_Output(MySQL_DBkey=MySQL_DBkey, 
                                                                              period_tableName=table
                                                                             )
        # do not append()
        concatenated_period_relevence = concatenated_period_relevence + period_relevence
        concatenated_period_hisCall = concatenated_period_hisCall + period_hisCall


    # convert to pd.dataframe and save to .csv

    # output concatenated_period_relevence
    OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
    data_file_name =  '../Outputs/' + csv_fileName + '_relevence.csv'
    Outputfilename = os.path.join(OutputfileDir, data_file_name) # ../ get back to upper level
    Outputfilename = os.path.abspath(os.path.realpath(Outputfilename))
    print Outputfilename
    
    results_R_pd = pd.DataFrame(concatenated_period_relevence)
    results_R_pd.to_csv(Outputfilename, index=False, header=False)

    # output concatenated_period_relevence
    OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
    data_file_name =  '../Outputs/' + csv_fileName + '_hisCall.csv'
    Outputfilename = os.path.join(OutputfileDir, data_file_name) # ../ get back to upper level
    Outputfilename = os.path.abspath(os.path.realpath(Outputfilename))
    print Outputfilename
    
    results_H_pd = pd.DataFrame(concatenated_period_hisCall)
    results_H_pd.to_csv(Outputfilename, index=False, header=False)

    ####################################################################
    return None


"""
####################################################################

UserUnique Functions of Phase3 Part1

####################################################################
"""



"""
####################################################################

# Execution of Phase2_Part1

####################################################################
"""

if __name__ == "__main__":

    ####################################################################

    MySQL_DBkey = {'host':'localhost', 'user':'', 'password':'', 'db':'','charset':'utf8'}

    keyword1 = 'trump'
    keyword2 = 'hillary'

    ####################################################################

    Start_Time = '2016-07-12 00:00:00'
    # Start_Time = '2016-10-14 00:00:00'
    Start_Time = pd.to_datetime(Start_Time)
    print "start time: ", Start_Time.strftime('_%Y_%m_%d_%H')

    End_Time = '2016-08-04 00:00:00'
    # End_Time = '2016-11-30 00:00:00'
    End_Time = pd.to_datetime(End_Time)
    print "end time: ", End_Time.strftime('_%Y_%m_%d_%H')
    
    # go to next time point
    # pin_time = pin_time + np.timedelta64(3600,'s')
    Time_Period = np.timedelta64(3600*24,'s') # one day
    
    header = 'tagunique_'
    
    ####################################################################
    
    TagUnique_Concatenate_Extract(MySQL_DBkey=MySQL_DBkey, 
                                  start_time=Start_Time, end_time=End_Time, 
                                  time_period=Time_Period, 
                                  csv_fileName='Tag_July',
                                  header='tagunique')


