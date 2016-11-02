
import numpy
from collections import OrderedDict
import os
import glob

import cPickle as pkl
from subprocess import Popen, PIPE

import sys
currdir = os.getcwd()
path = currdir + '/scripts/tokenizer'
print "path to DataSet_Pickle_Main: ", path
sys.path.insert(0, path)
from DataSet_Pickle import DataSet_Pickle_Train, DataSet_Pickle_Predict

'''
####################################################################
'''

def Tokenize_Main(dict_parameters, flag_trainOrpredict, 
                  dict_train, dict_test, flag_alsoTxt=False):

    # get current address
    currdir = os.getcwd()
    print 'currdir: ', currdir

    # path_preToken_DataSet = '../Data/DataSet_Tokenize/'
    path_preToken_DataSet = dict_parameters['dataset_path']
    path_preToken_DataSet = os.path.join(currdir, path_preToken_DataSet)
    print "path_preToken_DataSet: \n", path_preToken_DataSet

    # path_tokenizer = './scripts/tokenizer/'
    path_tokenizer = dict_parameters['tokenizer_path']
    path_tokenizer = os.path.join(currdir, path_tokenizer)
    print "path_tokenizer: \n", path_tokenizer

    # Data Pickle Operation
    if flag_trainOrpredict == True: 
        N_uniqueWords = DataSet_Pickle_Train(dict_parameters=dict_parameters,
                            DataSet_preToken_Path=path_preToken_DataSet,
                            path_tokenizer=path_tokenizer,
                            dict_train=dict_train, dict_test=dict_test)
    if flag_trainOrpredict == False: 
        N_uniqueWords = DataSet_Pickle_Predict(dict_parameters=dict_parameters,
                        DataSet_preToken_Path=path_preToken_DataSet,
                        path_tokenizer=path_tokenizer,
                        dict_dataset=dict_train)

    # return to currdir
    os.chdir(currdir)

    return N_uniqueWords

'''
####################################################################
'''


"""
########################################################################################
"""

if __name__ == '__main__':

    ####################################################################

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
        'lstm_saveto': 'lstm_model_trainAgainst_trump.npz',
        'lstm_loadfrom':'lstm_model_trainAgainst_trump.npz',
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
        'lstm_saveto': 'lstm_model_trainAgainst_trump.npz',
        'lstm_loadfrom':'lstm_model_trainAgainst_trump.npz',
        # LSTM model parameter save/load
        'Yvalue_list':['trump', 'hillary', 'neutral'],
        # root name for cases to be considered
        'trump_folder':['posi_neut', 'posi_neg'],
        'hillary_folder':['neut_posi', 'neg_posi'],
        'neutral_folder':['neg_neg'],
        
        'trump_score':2,
        'hillary_score':0,
        'neutral_score':1,
        }

    ####################################################################

    # tokenizer path variables
    path_preToken_Training = '../Data/DataSet_Training/'
    path_preToken_Predicting = '../Data/DataSet_Predicting/'
    
    path_tokenizer = './scripts/tokenizer/'

    # MarkedTag_Import file_name_list
    file_name_list = ['MarkedTag_keyword1.csv','MarkedTag_keyword2.csv']

    ####################################################################

    flag_training = False  
    flag_predicting = False  

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
        Tokenize_Main(dict_parameters = dict_tokenizeParameters_trainAgainst_hillary, 
                      flag_trainOrpredict = True )  
        
    
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
        dict_tokenizeParameters_trainAgainst_trump['posi_trump_folder'] = full_folder_list
        dict_tokenizeParameters_trainAgainst_trump['neg_trump_folder'] = []

        dict_tokenizeParameters_trainAgainst_hillary['posi_hillary_folder'] = full_folder_list
        dict_tokenizeParameters_trainAgainst_hillary['neg_hillary_folder'] = []

        dict_tokenizeParameters_trainAgainst_trumphillary['trump_folder'] = full_folder_list
        dict_tokenizeParameters_trainAgainst_trumphillary['hillary_folder'] = []
        dict_tokenizeParameters_trainAgainst_trumphillary['neutral_folder'] = []

        # load into list_dict_tokenizeParameters
        list_dict_tokenizeParameters = [dict_tokenizeParameters_trainAgainst_trump,
                                        dict_tokenizeParameters_trainAgainst_hillary,
                                        dict_tokenizeParameters_trainAgainst_trumphillary]

        Tokenize_Main(dict_parameters = dict_tokenizeParameters_trainAgainst_hillary, 
                      flag_trainOrpredict = False )     

