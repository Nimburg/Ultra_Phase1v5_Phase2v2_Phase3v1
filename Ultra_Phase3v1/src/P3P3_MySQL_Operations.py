

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

def NetworkedTags_TagUnique_Concatenate(MySQL_DBkey, list_tableNames):
    '''
    extract information from a given list of UserUnique_time tables
    extracted information: tagText, totalCall, score1_fin, Ncall1_fin, score2_fin, Ncall2_fin

    list_tableNames: list of table names within a set time-period
    '''
    # basic data structre
    # key as tagText
    Tags_Today_preIter = col.defaultdict(col.defaultdict)
    # key and val as: totalCall -> int, etc
    # Tags_Today_preIter['tag_example'] = col.defaultdict() 
    # 
    # Tags_Today_preIter['tag_example']['tagText'] = tagText
    # Tags_Today_preIter['tag_example']['score1_fin'] = numerical values
    # 
    # Tags_Today_preIter['tag_example']['tagScore1_text'] = list of scores_float
    # Tags_Today_preIter['tag_example']['tag_counter_text'] = dict()
    #                                                       key = tag, value = Ncall

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
SELECT tagText, totalCall, score1_fin, Ncall1_fin, score2_fin, Ncall2_fin, \
tagScore1_text, tagNcall1_text, tagScore2_text, tagNcall2_text, tag_counter_text
FROM %s
WHERE score1_fin >= 1.0 OR score2_fin >= 1.0
ORDER BY totalCall DESC"""
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
                    # numericals
                    totalCall = entry['totalCall']
                    score1_fin = entry['score1_fin']
                    Ncall1_fin = entry['Ncall1_fin']
                    score2_fin = entry['score2_fin']
                    Ncall2_fin = entry['Ncall2_fin']
                    # text values
                    tagScore1_text = str( entry['tagScore1_text'] ).decode('utf-8').encode('ascii', 'ignore') 
                    tagNcall1_text = str( entry['tagNcall1_text'] ).decode('utf-8').encode('ascii', 'ignore')
                    tagScore2_text = str( entry['tagScore2_text'] ).decode('utf-8').encode('ascii', 'ignore')
                    tagNcall2_text = str( entry['tagNcall2_text'] ).decode('utf-8').encode('ascii', 'ignore')

                    tag_counter_text = str( entry['tag_counter_text'] ).decode('utf-8').encode('ascii', 'ignore')

                    #####################
                    # split text values #
                    #####################

                    ####################################################################
                    def str_list_numeric(input_str, intOrfloat):
                        input_list_str = input_str.split(',')
                        input_res = []
                        try:
                            if intOrfloat == 'int': 
                                input_res = [ int(item) for item in input_list_str if len(item)>0 ]
                            if intOrfloat == 'float': 
                                # check against abnormal values, should truncation due to varichar(3000)
                                input_res = [ float(item) for item in input_list_str if len(item)>0 ]                           
                        finally:
                            pass
                        return input_res
                    ####################################################################

                    tagScore1_list_float = str_list_numeric(tagScore1_text, 'float')
                    tagNcall1_list_int = str_list_numeric(tagNcall1_text, 'int')
                    tagScore2_list_float = str_list_numeric(tagScore2_text, 'float')
                    tagNcall2_list_int = str_list_numeric(tagNcall2_text, 'int')

                    ####################################################################

                    tag_counter_list = tag_counter_text.split(',')
                    tag_counter_list_tuple = []
                    try:
                        tag_counter_list_tuple = [ ( item.split(':')[0], item.split(':')[1] ) for item in tag_counter_list if (':' in item)
                                                 ]
                    finally:
                        pass
                    # convert to int
                    for idx in range( len(tag_counter_list_tuple) ):
                        if len( tag_counter_list_tuple[idx][1] )>0: 
                            tag_counter_list_tuple[idx] = ( tag_counter_list_tuple[idx][0], int( tag_counter_list_tuple[idx][1] ) )
                        else:
                            tag_counter_list_tuple.remove( tag_counter_list_tuple[idx] )

                    #####################################
                    # load data into Tags_Today_preIter #
                    #####################################

                    # if tagText in Tags_Today_preIter
                    if tagText in Tags_Today_preIter:
                        # update tagText's values
                        # totalCall is a per-time_window variable, thus added up
                        Tags_Today_preIter[tagText]['totalCall'] = Tags_Today_preIter[tagText]['totalCall'] + totalCall
                        # always using the latest values for those over-time variables
                        Tags_Today_preIter[tagText]['score1_fin'] = score1_fin
                        Tags_Today_preIter[tagText]['Ncall1_fin'] = Ncall1_fin
                        Tags_Today_preIter[tagText]['score2_fin'] = score2_fin
                        Tags_Today_preIter[tagText]['Ncall2_fin'] = Ncall2_fin

                        # keep original text values, along with processed values
                        # new_str = old_str + ',' + coming_str
                        # new_list = old_list + coming_list
                        Tags_Today_preIter[tagText]['tagScore1_text'] = Tags_Today_preIter[tagText]['tagScore1_text'] + \
                                                                        ',' + tagScore1_text
                        Tags_Today_preIter[tagText]['tagScore1_list_float'] = Tags_Today_preIter[tagText]['tagScore1_list_float'] + \
                                                                                tagScore1_list_float

                        Tags_Today_preIter[tagText]['tagNcall1_text'] = Tags_Today_preIter[tagText]['tagNcall1_text'] + \
                                                                        ',' + tagNcall1_text
                        Tags_Today_preIter[tagText]['tagNcall1_list_int'] = Tags_Today_preIter[tagText]['tagNcall1_list_int'] +\
                                                                                tagNcall1_list_int

                        Tags_Today_preIter[tagText]['tagScore2_text'] = Tags_Today_preIter[tagText]['tagScore2_text'] +\
                                                                        ',' + tagScore2_text
                        Tags_Today_preIter[tagText]['tagScore2_list_float'] = Tags_Today_preIter[tagText]['tagScore2_list_float'] +\
                                                                                tagScore2_list_float

                        Tags_Today_preIter[tagText]['tagNcall2_text'] = Tags_Today_preIter[tagText]['tagNcall2_text'] +\
                                                                        ',' + tagNcall2_text
                        Tags_Today_preIter[tagText]['tagNcall2_list_int'] = Tags_Today_preIter[tagText]['tagNcall2_list_int'] +\
                                                                                tagNcall2_list_int

                        # tag_counter_text
                        Tags_Today_preIter[tagText]['tag_counter_text'] = Tags_Today_preIter[tagText]['tag_counter_text'] +\
                                                                        ',' + tag_counter_text
                        # Tags_Today_preIter[tagText]['tag_counter'] need to be updated
                        for item_tuple in tag_counter_list_tuple:
                            Tags_Today_preIter[tagText]['tag_counter'][item_tuple[0]] += item_tuple[1]

                    # if tagText not in Tags_Today_preIter
                    if tagText not in Tags_Today_preIter:
                        # add tagText as col.defaultdict()
                        Tags_Today_preIter[tagText] = col.defaultdict()
                        # add numerical values
                        Tags_Today_preIter[tagText]['totalCall'] = totalCall
                        Tags_Today_preIter[tagText]['score1_fin'] = score1_fin
                        Tags_Today_preIter[tagText]['Ncall1_fin'] = Ncall1_fin
                        Tags_Today_preIter[tagText]['score2_fin'] = score2_fin
                        Tags_Today_preIter[tagText]['Ncall2_fin'] = Ncall2_fin

                        # keep original text values, along with processed values
                        Tags_Today_preIter[tagText]['tagScore1_text'] = tagScore1_text
                        Tags_Today_preIter[tagText]['tagScore1_list_float'] = tagScore1_list_float

                        Tags_Today_preIter[tagText]['tagNcall1_text'] = tagNcall1_text
                        Tags_Today_preIter[tagText]['tagNcall1_list_int'] = tagNcall1_list_int

                        Tags_Today_preIter[tagText]['tagScore2_text'] = tagScore2_text
                        Tags_Today_preIter[tagText]['tagScore2_list_float'] = tagScore2_list_float

                        Tags_Today_preIter[tagText]['tagNcall2_text'] = tagNcall2_text
                        Tags_Today_preIter[tagText]['tagNcall2_list_int'] = tagNcall2_list_int

                        # dict for tag_counter_text
                        Tags_Today_preIter[tagText]['tag_counter_text'] = tag_counter_text
                        Tags_Today_preIter[tagText]['tag_counter'] = col.Counter()
                        for item_tuple in tag_counter_list_tuple:
                            # here += will initialize col.Counter()
                            Tags_Today_preIter[tagText]['tag_counter'][item_tuple[0]] += item_tuple[1]

                # end of for entry in result:
        finally:
            pass

    ####################################################################
    connection.close()
    return Tags_Today_preIter


'''
####################################################################
'''

def NetworkedTags_TagUnique_postIterSave(MySQL_DBkey, 
                                         New_tableName,
                                         Tags_Today_postIter):
    '''
    Tags_Today_postIter:
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
    
    ####################################################################
    # create New table
    # table Name
    tableName = New_tableName
    #Comd
    Comd_Unique_Concatenated = """
DROP TABLE IF EXISTS %s;
CREATE TABLE IF NOT EXISTS %s
(
    tagText varchar(255) PRIMARY KEY NOT NULL,
    senti_score1 float,
    senti_score2 float,
    totalCall int,
    score1_fin  float,
    Ncall1_fin int,
    score1_ave float,
    score1_std float,
    score1_median float,
    score1_max float,
    score1_min float,
    score2_fin float,
    Ncall2_fin int, 
    score2_ave float,
    score2_std float,
    score2_median float,
    score2_max float,
    score2_min float,
    tagScore1_text TEXT, 
    tagNcall1_text TEXT, 
    tagScore2_text TEXT, 
    tagNcall2_text TEXT, 
    tag_counter_text TEXT
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
    

    ####################################################################
    # go through Tags_Today_postIter
    print "start loading %s" % tableName    
    for tag in Tags_Today_postIter:
        # command for Tweet_Stack
        comd_TagUnique_Insert = """
INSERT INTO %s (tagText, senti_score1, senti_score2, totalCall, \
score1_fin, Ncall1_fin, score1_ave, score1_std, score1_median, score1_max, score1_min,\
score2_fin, Ncall2_fin, score2_ave, score2_std, score2_median, score2_max, score2_min,\
tagScore1_text, tagNcall1_text, tagScore2_text, tagNcall2_text, tag_counter_text)
VALUES ( '%s', %s, %s, %s, \
%s, %s, %s, %s, %s, %s, %s, \
%s, %s, %s, %s, %s, %s, %s, \
'%s', '%s', '%s', '%s', '%s')
ON DUPLICATE KEY UPDATE totalCall = %s;"""
        # execute commands
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_TagUnique_Insert % tuple( [tableName] + 
                                                               [tag] + 
                                                               [str(Tags_Today_postIter[tag]['senti_score1'])] + 
                                                               [str(Tags_Today_postIter[tag]['senti_score2'])] + 
                                                               [str(Tags_Today_postIter[tag]['totalCall'])] + 
                                                               [str(Tags_Today_postIter[tag]['score1_fin'])] + 
                                                               [str(Tags_Today_postIter[tag]['Ncall1_fin'])] + 
                                                               [str(Tags_Today_postIter[tag]['score1_ave'])] + 
                                                               [str(Tags_Today_postIter[tag]['score1_std'])] + 
                                                               [str(Tags_Today_postIter[tag]['score1_median'])] + 
                                                               [str(Tags_Today_postIter[tag]['score1_max'])] + 
                                                               [str(Tags_Today_postIter[tag]['score1_min'])] + 
                                                               [str(Tags_Today_postIter[tag]['score2_fin'])] + 
                                                               [str(Tags_Today_postIter[tag]['Ncall2_fin'])] + 
                                                               [str(Tags_Today_postIter[tag]['score2_ave'])] + 
                                                               [str(Tags_Today_postIter[tag]['score2_std'])] + 
                                                               [str(Tags_Today_postIter[tag]['score2_median'])] + 
                                                               [str(Tags_Today_postIter[tag]['score2_max'])] + 
                                                               [str(Tags_Today_postIter[tag]['score2_min'])] + 
                                                               [ Tags_Today_postIter[tag]['tagScore1_text'] ] + 
                                                               [ Tags_Today_postIter[tag]['tagNcall1_text'] ] + 
                                                               [ Tags_Today_postIter[tag]['tagScore2_text'] ] +                                                                
                                                               [ Tags_Today_postIter[tag]['tagNcall2_text'] ] + 
                                                               [ Tags_Today_postIter[tag]['tag_counter_text'] ] + 
                                                               [str(Tags_Today_postIter[tag]['totalCall'])]
                                                              ) 
                              )
            # commit commands
            connection.commit()
        finally:
            pass
    print "Done loading %s" % tableName

    ####################################################################
    connection.close()  
    return None


'''
####################################################################
'''

def NetworkedTags_Extract_RequestedTags(MySQL_DBkey, tag_text, postIter_taleName_list):
    '''
    postIter_taleName_list: list of postIter_taleName_list Names
    extract data from each table of postIter_taleName_list
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

    tag_perDay_results = []

    for table_name in postIter_taleName_list: 
        # NetworkedTags_TagUnique_2016_10_22_18
        date = table_name[24:-3]
        # Comd to extract data from each table
        Comd_Extract = """
SELECT senti_score1, senti_score2, totalCall, \
score1_fin, Ncall1_fin, score1_ave, score1_std, score1_median, score1_max, score1_min,\
score2_fin, Ncall2_fin, score2_ave, score2_std, score2_median, score2_max, score2_min
FROM %s
WHERE tagText = '%s';"""
        # Execute Comds
        print "extracting tweets from %s" % table_name
        try:
            with connection.cursor() as cursor:
                cursor.execute( Comd_Extract % tuple( [table_name]+[tag_text] )
                              )
                result = cursor.fetchall()
                # loop through all rows of this table
                holder_list = []
                for entry in result:
                    holder_list.append( date )

                    holder_list.append( entry['senti_score1'] )
                    holder_list.append( entry['senti_score2'] )
                    holder_list.append( entry['totalCall'] )

                    holder_list.append( entry['score1_fin'] )
                    holder_list.append( entry['Ncall1_fin'] )
                    holder_list.append( entry['score1_ave'] )
                    holder_list.append( entry['score1_std'] )
                    holder_list.append( entry['score1_median'] )
                    holder_list.append( entry['score1_max'] )
                    holder_list.append( entry['score1_min'] )
                    
                    holder_list.append( entry['score2_fin'] )
                    holder_list.append( entry['Ncall2_fin'] )
                    holder_list.append( entry['score2_ave'] )
                    holder_list.append( entry['score2_std'] )
                    holder_list.append( entry['score2_median'] )
                    holder_list.append( entry['score2_max'] )
                    holder_list.append( entry['score2_min'] )

                    # load into list of lists as: 
                    tag_perDay_results.append( holder_list )
        finally:
            pass

    ####################################################################
    # convert to pd.dataframe and output to tagText.csv

    # output tag_perDay_results
    OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
    data_file_name =  '../Outputs/' + tag_text + '.csv'
    Outputfilename = os.path.join(OutputfileDir, data_file_name) # ../ get back to upper level
    Outputfilename = os.path.abspath(os.path.realpath(Outputfilename))
    print Outputfilename
    
    # header
    csv_header = ['date','senti_score1','senti_score2','totalCall',\
                  'score1_fin','Ncall1_fin','score1_ave','score1_std','score1_median','score1_max','score1_min',\
                  'score2_fin','Ncall2_fin','score2_ave','score2_std','score2_median','score2_max','score2_min']

    tag_perDay_pd = pd.DataFrame(tag_perDay_results)
    tag_perDay_pd.to_csv(Outputfilename, index=False, header=csv_header)

    ####################################################################
    connection.close()
    return tag_perDay_results

'''
####################################################################
'''


















