

import json
import os
import numpy as np 
import pandas as pd
import collections as col
import statistics
import csv

import pymysql.cursors

import cPickle as pickle


'''
#####################################################################################################
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
#####################################################################################################
'''

def PopularTags_Dictionary(MySQL_DBkey, list_tableNames, 
                           size_perDayRanking):
    '''
    extract most used tags on per day basis
    aggregate into an ordered dictionary of hash tags for the entire period
    tag2index and index2tag

    size_perDayRanking: threshold for per-day tag usage
    list_tableNames: list of table names within a set time-period
    '''

    # key as tagText, val as accumulated usage
    TagUsage_dict = col.Counter()

    ####################################################################

    # Connect to the database
    connection = pymysql.connect(host=MySQL_DBkey['host'],
                                 user=MySQL_DBkey['user'],
                                 password=MySQL_DBkey['password'],
                                 db=MySQL_DBkey['db'],
                                 charset=MySQL_DBkey['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)
    
    ####################################################################

    # go through list_tableNames
    for tableName in list_tableNames:
        # Comd to extract data from each table
        Comd_Extract = """
SELECT tagText, totalCall
FROM %s
ORDER BY totalCall DESC
LIMIT %i"""
        # Execute Comds
        print "extracting tags' usage from %s" % tableName
        try:
            with connection.cursor() as cursor:
                cursor.execute( Comd_Extract % (tableName, size_perDayRanking) )
                result = cursor.fetchall()
                # loop through all rows of this table
                for entry in result:
                    # {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
                    # in MySQL, tag format is utf8, but in raw data as ASCII
                    tagText = str( entry['tagText'] ).decode('utf-8').encode('ascii', 'ignore')             
                    # numericals
                    totalCall = entry['totalCall']
                    # TagUsage_dict = col.Counter()
                    TagUsage_dict[tagText] += totalCall
                # end of for entry in result:
        finally:
            pass
        # check
        print "Number of Tags in TagUsage_dict: %i" % len(TagUsage_dict)
    # end of for tableName in list_tableNames:

    ####################################################################

    ##################################################
    # create tag2index as dict[tag] = (index, usage) #
    # create index2tag as list[(tag, usage)]         # 
    ##################################################
    
    # create ordered tuple list of (tag, usage)
    tag_tuple_list = []
    for key in TagUsage_dict:
        tag_tuple_list.append( (key, TagUsage_dict[key]) )
    # sort by usage in descent
    tag_tuple_list = sorted( tag_tuple_list, key=lambda x: x[1])
    index2tag = [ tag_tuple_list[-i-1] for i in range(len(tag_tuple_list)) ]
    # tag2index
    tag2index = dict()
    for idx in range(len(index2tag)):
        tag2index[ index2tag[idx][0] ] = (idx, index2tag[idx][1])

    ###########################
    # save tag/index as .csv #
    ###########################

    tag_index_list = []
    # save as [tag, idx, usage]
    for idx in range(len(index2tag)):
        tag_index_list.append( [ index2tag[idx][0], idx, index2tag[idx][1] ] )

    csv_fileName = 'word_index_usage.csv'
    # output csv
    OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
    data_file_name =  '../Outputs/' + csv_fileName
    Outputfilename = os.path.join(OutputfileDir, data_file_name) # ../ get back to upper level
    Outputfilename = os.path.abspath(os.path.realpath(Outputfilename))
    print Outputfilename
    
    tag_index_pd = pd.DataFrame(tag_index_list)
    tag_index_pd.to_csv(Outputfilename, index=False, header=False)
    print "Done saving tag_index_list into %s" % csv_fileName

    ####################################################################
    connection.close()
    return index2tag, tag2index

'''
#####################################################################################################
'''

def DataPrep_perDayTweets(MySQL_DBkey, tableName, 
                          index2tag, tag2index,
                          train_test_ratio,
                          size_tagDictionary=None,
                          flag_trainOrpredict=True, flag_forcePredict=True,
                          DataSet_header=None):
    '''
    this function operate on the basis of EACH single scoredfiltered_tweets table
    converts post-comparison sentiment of tweet into Y value
    converts taglist into X vector
    pickle the result

    size_tagDictionary: size of the tag2index dictionary to use
                        ALSO effectively the dimension of X_vector !!!

    tag2index as dict[tag] = (index, usage)
    index2tag as list[(tag, usage)]  
    DataSet_header: header name of this data set
    flag_trainOrpredict: Ture as training, False as predicting
    '''
    assert size_tagDictionary is not None
    assert DataSet_header is not None

    ####################################################################

    # Connect to the database
    connection = pymysql.connect(host=MySQL_DBkey['host'],
                                 user=MySQL_DBkey['user'],
                                 password=MySQL_DBkey['password'],
                                 db=MySQL_DBkey['db'],
                                 charset=MySQL_DBkey['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)
    
    ####################################################################

    ###########################
    # get full list of Tweets #
    # and their X,Y values    #
    ###########################
    
    # list of [tweetID_str, Y_val, X_vector]
    Tweet_XY_list = []

    # Comd to extract data from each table
    Comd_Extract = """
SELECT tweetID, taglist, HMS_NW_senti_score1, HMS_NW_senti_score2
FROM %s
WHERE HMS_NW_senti_score1 != HMS_NW_senti_score2"""
    # Execute Comds
    print "extracting Tweets from %s" % tableName
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_Extract % tableName )
            result = cursor.fetchall()
            # loop through all rows of this table
            for entry in result:
                # text
                taglist = str( entry['taglist'] ).decode('utf-8').encode('ascii', 'ignore')
                taglist = taglist.split(',')            
                # numericals
                senti_1 = entry['HMS_NW_senti_score1']
                senti_2 = entry['HMS_NW_senti_score2']
                # convert big_int to string
                tweetID_str = str( entry['tweetID'] )

                # calculate Y_val; 0 for hillary, 1 for trump
                Y_val = 0
                if senti_1 > senti_2:
                    Y_val = 1
                # calculate X_vector
                X_vector = [0]*size_tagDictionary
                flag_inDict = False
                for tag in taglist:
                    # tag2index as dict[tag] = (index, usage)
                    if (tag in tag2index) and (tag2index[tag][0] <= size_tagDictionary):
                        flag_inDict = True
                        # flag corresponding element of the size_tagDictionary-dimension X_vector as 1
                        X_vector[ tag2index[tag][0] ] = 1

                # append into Tweet_XY_list
                if flag_inDict == True: 
                    Tweet_XY_list.append( [tweetID_str, Y_val, X_vector] )
            # end of for entry in result:
    finally:
        pass
    connection.close()
    ####################################################################

    ###########################
    # shuffle Tweet_XY_list   #
    # split into train/test   #
    ###########################

    # train and test data set
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_full = []
    Y_full = []
    tweetID_full =[]
    # shuffle data set  
    index_list = np.arange(len(Tweet_XY_list))
    np.random.shuffle(index_list)
    index_pin = train_test_ratio*len(Tweet_XY_list)

    # load train/test data set
    # [tweetID_str, Y_val, X_vector]
    for idx in range(len(Tweet_XY_list)):
        # training
        if idx <= index_pin:
            X_train.append( Tweet_XY_list[ index_list[idx] ][2] )
            Y_train.append( Tweet_XY_list[ index_list[idx] ][1] )
        # testing
        if idx > index_pin:
            X_test.append( Tweet_XY_list[ index_list[idx] ][2] )
            Y_test.append( Tweet_XY_list[ index_list[idx] ][1] )            
    # full list
    # as ordered
    for idx in range(len(Tweet_XY_list)):
        X_full.append( Tweet_XY_list[idx][2] )
        Y_full.append( Tweet_XY_list[idx][1] )
        tweetID_full.append( Tweet_XY_list[idx][0] )

    ##################
    # pickle outputs #
    ##################

    # tableName as scoredfiltered_tweets_2016_11_08_00

    # as training
    if flag_trainOrpredict == True: 
        # outputs address
        OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
        OutputfileDir = os.path.join(OutputfileDir, '../Outputs/')
        # 2 * pkl.dump()
        fileName = OutputfileDir + DataSet_header + tableName[-14:-3] + '_train.pkl'
        f = open(fileName, 'wb')
        # pickle training set
        pickle.dump((X_train, Y_train), f, -1)
        # pickle testing set
        pickle.dump((X_test, Y_test), f, -1)
        f.close()

    # pickle the entire data set, used for training-prediction check
    if flag_forcePredict == True: 
        # outputs address
        OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
        OutputfileDir = os.path.join(OutputfileDir, '../Outputs/')
        # 2 * pkl.dump()
        fileName = OutputfileDir + DataSet_header + tableName[-14:-3] + '_PredictCheck.pkl'
        f = open(fileName, 'wb')
        # pickle full data set
        pickle.dump((X_full, Y_full), f, -1)
        # pickle tweetID
        pickle.dump(tweetID_full, f, -1)
        f.close()

    ####################################################################
    return None


