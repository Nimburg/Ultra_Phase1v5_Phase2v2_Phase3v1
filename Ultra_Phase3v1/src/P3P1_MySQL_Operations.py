

import json
import os
import numpy as np 
import pandas as pd
import collections as col

import pymysql.cursors


'''
####################################################################
'''

def Get_Table_Names(MySQL_DBkey, start_time, end_time, header):
    '''
    get table names

    header: basic table names, e.g. userunique or tagunique
    start_time: pd.to_datetime( time_string)
    end_time: 
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
    list_table_names = []
    pin_time = start_time
    # loop through time
    while (pin_time < end_time):
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
            list_table_names.append(table_name)
        else:
            print "%s does NOT exist." % table_name
        # go to next time point
        pin_time = pin_time + np.timedelta64(3600,'s')

    # end of loop
    connection.close()
    return list_table_names

'''
####################################################################
'''

def TagUnique_Concatenate(MySQL_DBkey, list_tableNames, New_tableName):
    '''
    extract information from a given list of UserUnique_time tables
    extracted information: tagText, totalCall, score1_fin, Ncall1_fin, score2_fin, Ncall2_fin

    list_tableNames: list of table names within a set time-period
    '''
    # basic data structre
    # key as tagText
    defaultdict_All_Tags = col.defaultdict(dict)
    # key and val as: totalCall -> int, etc
    # defaultdict_All_Tags['tag_example'] = dict() 

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
    # go through list_tableNames
    for tableName in list_tableNames:
        # Comd to extract data from each table
        Comd_Extract = """
SELECT tagText, totalCall, score1_fin, Ncall1_fin, score2_fin, Ncall2_fin 
FROM %s
WHERE score1_fin >= 1.0 AND score2_fin >= 1.0 AND Ncall1_fin >= 1000 AND Ncall2_fin >= 1000
ORDER BY totalCall DESC;"""
        # Execute Comds
        print "extracting tweets from %s" % tableName
        try:
            with connection.cursor() as cursor:
                cursor.execute( Comd_Extract % tableName )
                result = cursor.fetchall()
                # loop through all rows of this table
                for entry in result:
                    # {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
                    # in MySQL, tag format is utf8, but in raw data as ASCII
                    tagText = str( entry['tagText'] ).decode('utf-8').encode('ascii', 'ignore')             
                    totalCall = entry['totalCall']
                    score1_fin = entry['score1_fin']
                    Ncall1_fin = entry['Ncall1_fin']
                    score2_fin = entry['score2_fin']
                    Ncall2_fin = entry['Ncall2_fin']
                    # add results for this tag into defaultdict_All_Tags
                    if tagText in defaultdict_All_Tags:
                        # update tagText's values
                        # totalCall is a per-time_window variable, thus added up
                        defaultdict_All_Tags[tagText]['totalCall'] = defaultdict_All_Tags[tagText]['totalCall'] + totalCall
                        # always using the latest values for those over-time variables
                        defaultdict_All_Tags[tagText]['score1_fin'] = score1_fin
                        defaultdict_All_Tags[tagText]['Ncall1_fin'] = Ncall1_fin
                        defaultdict_All_Tags[tagText]['score2_fin'] = score2_fin
                        defaultdict_All_Tags[tagText]['Ncall2_fin'] = Ncall2_fin

                    if tagText not in defaultdict_All_Tags:
                        # add tagText
                        defaultdict_All_Tags[tagText] = dict()
                        # add tagText's values
                        defaultdict_All_Tags[tagText]['totalCall'] = totalCall
                        defaultdict_All_Tags[tagText]['score1_fin'] = score1_fin
                        defaultdict_All_Tags[tagText]['Ncall1_fin'] = Ncall1_fin
                        defaultdict_All_Tags[tagText]['score2_fin'] = score2_fin
                        defaultdict_All_Tags[tagText]['Ncall2_fin'] = Ncall2_fin

        finally:
            pass

    ####################################################################

    ##################################################################
    # concatenante a new UserUnique table using defaultdict_All_Tags #
    ##################################################################

    # create New table
    # table Name
    tableName = New_tableName
    #Comd
    Comd_Unique_Concatenated = """
DROP TABLE IF EXISTS %s;
CREATE TABLE IF NOT EXISTS %s
(
    tagText varchar(255) PRIMARY KEY NOT NULL,
    totalCall int NOT NULL,
    score1_fin float,
    Ncall1_fin int,
    score2_fin float,
    Ncall2_fin int
)ENGINE=InnoDB;"""
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_Unique_Concatenated % tuple ( [tableName]+[tableName] )
                          )
        # commit commands
        print tableName+" Initialized"
        connection.commit()
    finally:
        pass

    # go through defaultdict_All_Tags
    print "start loading %s" % tableName    
    for tag in defaultdict_All_Tags:
        # command for Tweet_Stack
        comd_TagUnique_Insert = """
INSERT INTO %s (tagText, totalCall, score1_fin, Ncall1_fin, score2_fin, Ncall2_fin)
VALUES ( '%s', %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE totalCall = %s;"""
        # execute commands
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_TagUnique_Insert % tuple( [tableName] + 
                                                               [tag] + 
                                                               [str(defaultdict_All_Tags[tag]['totalCall'])] + 
                                                               [str(defaultdict_All_Tags[tag]['score1_fin'])] + 
                                                               [str(defaultdict_All_Tags[tag]['Ncall1_fin'])] + 
                                                               [str(defaultdict_All_Tags[tag]['score2_fin'])] + 
                                                               [str(defaultdict_All_Tags[tag]['Ncall2_fin'])] + 
                                                               [str(defaultdict_All_Tags[tag]['totalCall'])]
                                                              ) 
                              )
            # commit commands
            connection.commit()
        finally:
            pass
    print "Done loading %s" % tableName

    ####################################################################
    connection.close()
    return defaultdict_All_Tags

'''
####################################################################
'''

def TagUnique_postConcatenation_Output(MySQL_DBkey, period_tableName):
    '''
    extract data from period_tableName
    results order by totalCall in decreasing order limit to top10

    for each line on .csv
    'date', 'tagText @ totalCall', 'relevence_to_key1', 'relevence_to_key2'
    or
    'date', 'tagText @ totalCall', 'hitoricNcall_to_key1', 'hitoricNcall_to_key2'

    into 2 different list of lists, output
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

    list_period_relevence = []
    list_period_hisCall = []

    # concatenated_Tags_2016_10_22_18
    date = period_tableName[18:-3]

    # Comd to extract data from each table
    Comd_Extract = """
SELECT tagText, totalCall, score1_fin, Ncall1_fin, score2_fin, Ncall2_fin 
FROM %s
ORDER BY totalCall DESC
LIMIT 10;"""
    # Execute Comds
    print "extracting tweets from %s" % period_tableName
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_Extract % period_tableName )
            result = cursor.fetchall()
            # loop through all rows of this table
            for entry in result:
                # {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
                # in MySQL, tag format is utf8, but in raw data as ASCII
                tagText = str( entry['tagText'] ).decode('utf-8').encode('ascii', 'ignore')             
                totalCall = entry['totalCall']
                
                score1_fin = entry['score1_fin']
                relevence_1 = "%.2f" % score1_fin
                Ncall1_fin = 1.0*entry['Ncall1_fin']/1000
                Ncall1_pK = "%.2f" % Ncall1_fin

                score2_fin = entry['score2_fin']
                relevence_2 = "%.2f" % score2_fin
                Ncall2_fin = 1.0*entry['Ncall2_fin']/1000
                Ncall2_pK = "%.2f" % Ncall2_fin

                # load into list of lists as: 
                # 'date', 'tagText @ totalCall', 'relevence_to_key1', 'relevence_to_key2'
                label = str(totalCall) + ' @ ' + tagText
                list_period_relevence.append( [date, label, relevence_1, relevence_2]
                                            )

                # load into list of lists as: 
                # 'date', 'tagText @ totalCall', 'hitoricNcall_to_key1', 'hitoricNcall_to_key2'
                list_period_hisCall.append( [date, label, Ncall1_pK, Ncall2_pK]
                                            )
    finally:
        pass

    ####################################################################
    # reverse order of list_period_relevence, list_period_hisCall 
    # due to graphing issues
    new_list = []
    new_list = [ list_period_relevence[-idx-1] for idx in range( len(list_period_relevence) ) ]
    list_period_relevence = new_list

    new_list = []
    new_list = [ list_period_hisCall[-idx-1] for idx in range( len(list_period_hisCall) ) ]
    list_period_hisCall = new_list

    return list_period_relevence, list_period_hisCall

"""
####################################################################

# test code 

####################################################################
"""




