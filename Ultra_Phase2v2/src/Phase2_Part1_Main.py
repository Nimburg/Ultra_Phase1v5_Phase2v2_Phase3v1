
####################################################################
# 
# Phase2_Part1 Design
# 
# 1. marking tags
# extract key_tags and relev_tags of both keywords
# 
# apply filters to text message:
# mentioned_users, https, ratio of non_tag_words and tag_words, number of non_tag_words
# create a filtered_tweet_stack
# use filtered_tweet_stack to try to flag those most frequently called tags
# 
# 
# filtered_tweet_stack should:
# use numerical Primary Key 
# has: tweetID, tweetTime, userID, tweet_text, taglist_text
# score1&2_by_tags, score1&2_by different LSTM models
# 
# 
# 
# 2. import the marked tags 
# and use the marked tags to export tweet text as training/testing data sets
# separating into groups of: key1 is trump, key2 is hillary
# 
# posi_posi, posi_neut, posi_neg
# neut_posi, neut_neut, neut_neg
# neg_posi, neg_neut, neg_neg
# 
# extract tweets into 9 folders.... as text file with tweetID as file name
# Functions in MarkedTag_Import.py would create all those 9 folders
# 
# apply filters to text message:
# mentioned_users, https, ratio of non_tag_words and tag_words, number of non_tag_words
# 
# Note: 
# 
# neut_neut has the largest number, in 10K order
# neg_neut and neut_neg are in order of several K
# anything with posi is in order of hundreds
# 
####################################################################
# 
# LSTM training can be separated into 3 groups:
# 
# 1st, training against 'trump' only
# with (posi_neut, posi_neg) score +1
# with (neg_posi, neg_neut, neg_neg) score 0
# 
# 2nd, training against 'hillary' only
# with (neut_posi, neg_posi) score +1
# with (posi_neg, neut_neg, neg_neg) score 0
# 
# 3rd, training against both keywords
# with (posi_neut, posi_neg) score +2
# with (neut_posi, neg_posi) score 0
# with (neg_neg) score +1
# 
# Thus, 3 LSTM networks (parameter set) will be trained !!!
# 
# Best training methods demands a balanced corpus !!!
# 
####################################################################


import json
import os
import numpy as np 
import pandas as pd
import collections as col

import pymysql.cursors

from MarkedTag_Import import MarkedTag_Import, TextExtract_byTags
from Tokenize_Main import Tokenize_Main


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

Main Function of Phase2_Part1

####################################################################
"""

def Pahse2_Part1_Main(file_name_list, MySQL_DBkey,
                      path_DataSet_training, path_tokenizer, ratio_train_test, 
                      size_dataset, thre_nonTagWords,
                      list_dict_tokenizeParameters, 
                      flag_trainOrpredict, flag_ridTags, flag_NeutralFiles, predict_Name=None):
    '''
        Phase2 Part1 Main Function
        tokenization and related

        Parameters
        ----------
        size_dataset: total number of tweets for tokenize
        ratio_train_test: ratio of train/test
        thre_nonTagWords: the threshold for number of non-tag words, 
                          above which a tweet is selected for tokenize
        Returns
        -------
    '''

    # set in-RAM table size
    Set_TempTable_Variables(MySQL_DBkey = MySQL_DBkey, N_GB = 4)

    ####################################################################

    # Connect to the database
    connection = pymysql.connect(host=MySQL_DBkey['host'],
                                 user=MySQL_DBkey['user'],
                                 password=MySQL_DBkey['password'],
                                 db=MySQL_DBkey['db'],
                                 charset=MySQL_DBkey['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)

    ####################################################################
    # This is for EITHER training OR predicting LSTM purposes !!!

    # create MarkedTags_dict from .csv
    MarkedTags_dict = MarkedTag_Import(file_name_list=file_name_list)
    
    if flag_trainOrpredict == True: 
        SQL_tableName = "training"
    if flag_trainOrpredict == False and predict_Name is not None: 
        SQL_tableName = predict_Name

    # extract all tweets from tweet_stack which contains key_tags
    # using MarkedTags_dict to create
    dict_train, dict_test = TextExtract_byTags(connection=connection, MarkedTags_dict=MarkedTags_dict, 
                                               path_save=path_DataSet_training, flag_trainOrpredict=flag_trainOrpredict, # training LSTM
                                               ratio_train_test=ratio_train_test, size_dataset=size_dataset, 
                                               thre_nonTagWords=thre_nonTagWords, 
                                               flag_ridTags=flag_ridTags, flag_NeutralFiles=flag_NeutralFiles, 
                                               SQL_tableName=SQL_tableName)
    
    # tokenization
    for dict_case in list_dict_tokenizeParameters: 
        dict_case['N_uniqueWords'] = Tokenize_Main(dict_parameters = dict_case, 
                                                   flag_trainOrpredict=flag_trainOrpredict,
                                                   dict_train=dict_train, dict_test=dict_test)
        print dict_case

    ####################################################################
    return None


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
    # dict_parameters for training

    dict_tokenizeParameters_trainAgainst_trump = {
        'dataset':'trainAgainst_trump', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Training/',
        'tokenizer_path': './scripts/tokenizer/',
        # same for all cases
        'lstm_saveto': 'lstm_model_trainAgainst_trump.npz',
        'lstm_loadfrom':'lstm_model_trainAgainst_trump.npz',
        # LSTM model parameter save/load
        'Yvalue_list':['posi_trump', 'neg_trump'],
        # root name for cases to be considered
        'posi_trump_folder':['posi_neut', 'posi_neg'],
        'neg_trump_folder':['neg_posi', 'neg_neut', 'neg_neg'],
        
        'posi_trump_score':1,
        'neg_trump_score':0
        }

    dict_tokenizeParameters_trainAgainst_hillary = {
        'dataset':'trainAgainst_hillary', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Training/',
        'tokenizer_path': './scripts/tokenizer/',
        # same for all cases
        'lstm_saveto': 'lstm_model_trainAgainst_hillar.npz',
        'lstm_loadfrom':'lstm_model_trainAgainst_hillar.npz',
        # LSTM model parameter save/load
        'Yvalue_list':['posi_hillary', 'neg_hillary'],
        # root name for cases to be considered
        'posi_hillary_folder':['neut_posi', 'neg_posi'],
        'neg_hillary_folder':['posi_neg', 'neut_neg', 'neg_neg'],
        
        'posi_hillary_score':1,
        'neg_hillary_score':0
        }

    dict_tokenizeParameters_trainAgainst_trumphillary = {
        'dataset':'trainAgainst_trumphillary', 
        # PLUS .pkl or dict.pkl for LSTM
        'dataset_path': '../Data/DataSet_Training/',
        'tokenizer_path': './scripts/tokenizer/',
        # same for all cases
        'lstm_saveto': 'lstm_model_trainAgainst_trumphillar.npz',
        'lstm_loadfrom':'lstm_model_trainAgainst_trumphillar.npz',
        # LSTM model parameter save/load
        'Yvalue_list':['trump', 'hillary', 'neutral'],
        # root name for cases to be considered
        'trump_folder':['posi_neut', 'posi_neg'],
        'hillary_folder':['neut_posi', 'neg_posi'],
        'neutral_folder':['neg_neg', 'neut_neg', 'neg_neut', 'neut_neut'],
        
        'trump_score':2,
        'hillary_score':0,
        'neutral_score':1,
        }

    ####################################################################
    path_tokenizer = './scripts/tokenizer/'
    
    # training
    path_preToken_Training = '../Data/DataSet_Training/'
    file_name_list_training = ['MarkedTag_keyword1.csv','MarkedTag_keyword2.csv']   
    flag_training = False 

    # predicting
    path_preToken_Predicting = '../Data/DataSet_Predicting/'
    file_name_list_predicting = ['MarkedTag_keyword1.csv','MarkedTag_keyword2.csv']
    predict_Name = "Nword5000_Dim1024"
    flag_predicting = True 

    ####################################################################
    # for training !!!
    if flag_training == True:

        dict_tokenizeParameters_trainAgainst_trump['dataset_path'] = path_preToken_Training
        dict_tokenizeParameters_trainAgainst_hillary['dataset_path'] = path_preToken_Training
        dict_tokenizeParameters_trainAgainst_trumphillary['dataset_path'] = path_preToken_Training

        # load into list_dict_tokenizeParameters
        list_dict_tokenizeParameters = [dict_tokenizeParameters_trainAgainst_trump,
                                        dict_tokenizeParameters_trainAgainst_hillary,
                                        dict_tokenizeParameters_trainAgainst_trumphillary]

        Pahse2_Part1_Main(file_name_list=file_name_list_training, MySQL_DBkey=MySQL_DBkey, 
                          path_DataSet_training=path_preToken_Training, path_tokenizer=path_tokenizer,
                          ratio_train_test=0.8, 
                          size_dataset=None, # total number of tweets for tokenize
                          thre_nonTagWords=10, 
                          # the threshold for number of non-tag words, 
                          # above which a tweet is selected for tokenize
                          list_dict_tokenizeParameters=list_dict_tokenizeParameters,
                          # list of dicts 
                          # each dict contains parameters for tokenization for specific cases
                          # related to corresponding LSTM training
                          flag_trainOrpredict=True, # training
                          flag_ridTags=False , flag_NeutralFiles=True 
                          )
    
    ####################################################################
    # for predicting !!!
    if flag_predicting == True:

        # setting path for .txt files
        dict_tokenizeParameters_trainAgainst_trump['dataset_path'] = path_preToken_Predicting
        dict_tokenizeParameters_trainAgainst_hillary['dataset_path'] = path_preToken_Predicting
        dict_tokenizeParameters_trainAgainst_trumphillary['dataset_path'] = path_preToken_Predicting

        # setting correct 9 folders to the class with highest Y-value
        full_folder_list = ['posi_posi', 'posi_neut', 'posi_neg',
                            'neut_posi', 'neut_neut', 'neut_neg',
                            'neg_posi', 'neg_neut', 'neg_neg']
        # thus passing the Y-value_max into LSTM
        # and setting all other folders to [], avoiding overlapping data
        # dict_tokenizeParameters_trainAgainst_trump['posi_trump_folder'] = full_folder_list
        # dict_tokenizeParameters_trainAgainst_trump['neg_trump_folder'] = []

        # load into list_dict_tokenizeParameters
        list_dict_tokenizeParameters = [dict_tokenizeParameters_trainAgainst_trump,
                                        dict_tokenizeParameters_trainAgainst_hillary,
                                        dict_tokenizeParameters_trainAgainst_trumphillary]

        Pahse2_Part1_Main(file_name_list=file_name_list_predicting, MySQL_DBkey=MySQL_DBkey, 
                          path_DataSet_training=path_preToken_Predicting, path_tokenizer=path_tokenizer,
                          ratio_train_test=0.8, 
                          size_dataset=None, # total number of tweets for tokenize
                          thre_nonTagWords=10, 
                          # the threshold for number of non-tag words, 
                          # above which a tweet is selected for tokenize
                          list_dict_tokenizeParameters=list_dict_tokenizeParameters,
                          predict_Name=predict_Name,
                          # list of dicts 
                          # each dict contains parameters for tokenization for specific cases
                          # related to corresponding LSTM training
                          flag_trainOrpredict=False, 
                          flag_ridTags=False , flag_NeutralFiles=True 
                          )




