

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
import six.moves.cPickle as pickle


"""
####################################################################
"""

def NLTK_Tokenize_DataPickle(Data_Train, Data_Test=None, Data_Full=None, 
                             DataSet_Name='ScoredTweets', 
                             flag_trainOrpredict=True, flag_forcePredict=True,
                             vocabulary_size=1000):
    '''
    Data_Train: training data, [tweetID_str, tweetTime_str, tweetText, senti_flag]
                tweetText has already been decoded, and all in lower case
    
    DataSet_Name: header name of this data set
    flag_trainOrpredict: Ture as training, False as predicting
    vocabulary_size: size of word2index dict
    '''

    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    ####################################################################

    #########################
    # tokenize Full DataSet #
    #########################

    # get tweetID and senti_flag
    # [tweetID_str, tweetTime_str, tweetText, senti_flag]
    tweetID_Full_list = [ tweet[0] for tweet in Data_Full ]
    senti_Full_list = [ tweet[3] for tweet in Data_Full ]

    # with sentence_start/end
    sentences = [ tweet[2] for tweet in Data_Full]
    print "Parsed %d sentences." % (len(sentences))
    tokenized_sentences = [ nltk.word_tokenize(sent) for sent in sentences ]

    ##########################
    # create word2index dict #
    ##########################
    
    # Count the word frequencies
    word_freq = nltk.FreqDist( itertools.chain(*tokenized_sentences) )
    print "Found %d unique words tokens." % len(word_freq.items())
    vocab = word_freq.most_common(vocabulary_size-1) 
    
    index_to_word = [word[0] for word in vocab]
    index_to_word = [unknown_token] + index_to_word
    
    # word_to_index is a dict of key=word, val=index
    word_to_index = dict( [(w,i) for i,w in enumerate(index_to_word)] )
    
    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    ###########################
    # word2index full dataset #
    ###########################

    # for checking
    tokenized_sentences_old = tokenized_sentences

    # Replace all words not in our vocabulary with the unknown token
    new_tokenized_sentences = []
    for sent in tokenized_sentences:
        new_tokenized_sentences.append( [ word_to_index[w] if w in word_to_index else word_to_index[unknown_token] for w in sent] )
    tokenized_sentences = new_tokenized_sentences

    # check Y_val = 1
    counter = 0
    for idx in range(len(tokenized_sentences)):
        if senti_Full_list[idx] == 1:
            print "Y_val == 1"
            print tokenized_sentences_old[idx]
            print tokenized_sentences[idx]
            print senti_Full_list[idx]
            print tweetID_Full_list[idx]
            counter+=1
        if counter > 5:
            break
    # check Y_val = 0
    counter = 0
    for idx in range(len(tokenized_sentences)):
        if senti_Full_list[idx] == 0:
            print "Y_val == 0"
            print tokenized_sentences_old[idx]
            print tokenized_sentences[idx]
            print senti_Full_list[idx]
            print tweetID_Full_list[idx]
            counter+=1
        if counter > 5:
            break   

    ####################################################################

    ###########################
    # save word2index as .csv #
    ###########################

    word2index_list = []
    # through dict
    for word in word_to_index:
        word2index_list.append( [word, word_to_index[word]] )

    csv_fileName = DataSet_Name + '_word2index.csv'
    # output csv
    OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
    data_file_name =  '../Data/' + csv_fileName
    Outputfilename = os.path.join(OutputfileDir, data_file_name) # ../ get back to upper level
    Outputfilename = os.path.abspath(os.path.realpath(Outputfilename))
    print Outputfilename
    
    word2index_pd = pd.DataFrame(word2index_list)
    word2index_pd.to_csv(Outputfilename, index=False, header=False)
    print "Done saving word_to_index into %s" % csv_fileName

    ####################################################################

    #######################
    # tokenize Data_Train #
    #######################

    # get tweetID and senti_flag
    tweetID_training_list = [ tweet[0] for tweet in Data_Train ]
    senti_training_list = [ tweet[3] for tweet in Data_Train ]

    # tokenize training sentences
    sentences = ["%s %s %s" % (sentence_start_token, tweet[2], sentence_end_token) for tweet in Data_Train]
    print "Parsed %d sentences for training." % (len(sentences))
    tokenized_training = [ nltk.word_tokenize(sent) for sent in sentences ]

    new_tokenized_sentences = []
    for sent in tokenized_training:
        new_tokenized_sentences.append( [ word_to_index[w] if w in word_to_index else word_to_index[unknown_token] for w in sent] )
    tokenized_training = new_tokenized_sentences

    ####################################################################

    ######################
    # tokenize Data_Test #
    ######################

    if Data_Test is not None:
        # get tweetID and senti_flag
        tweetID_testing_list = [ tweet[0] for tweet in Data_Test ]
        senti_testing_list = [ tweet[3] for tweet in Data_Test ]

        # tokenize testing sentences
        sentences = ["%s %s %s" % (sentence_start_token, tweet[2], sentence_end_token) for tweet in Data_Test]
        print "Parsed %d sentences for testing." % (len(sentences))
        tokenized_testing = [ nltk.word_tokenize(sent) for sent in sentences ]

        new_tokenized_sentences = []
        for sent in tokenized_testing:
            new_tokenized_sentences.append( [ word_to_index[w] if w in word_to_index else word_to_index[unknown_token] for w in sent] )
        tokenized_testing = new_tokenized_sentences

    ###############################################################################
    
    ##################
    # pickle outputs #
    ##################

    # as training
    if flag_trainOrpredict == True: 
        # outputs address
        OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
        OutputfileDir = os.path.join(OutputfileDir, '../Data/DataSet_Training/')
        # 2 * pkl.dump()
        imdb_pkl_fileName = OutputfileDir + DataSet_Name + '_train.pkl'
        f = open(imdb_pkl_fileName, 'wb')
        # pickle training set
        pickle.dump((tokenized_training, senti_training_list), f, -1)
        # pickle testing set
        pickle.dump((tokenized_testing, senti_testing_list), f, -1)
        f.close()

    # pickle the entire data set, used for training-prediction check
    if flag_forcePredict == True: 
        # outputs address
        OutputfileDir = os.path.dirname(os.path.realpath('__file__'))
        OutputfileDir = os.path.join(OutputfileDir, '../Data/DataSet_Predicting/')
        # 2 * pkl.dump()
        imdb_pkl_fileName = OutputfileDir + DataSet_Name + '_PredictCheck.pkl'
        f = open(imdb_pkl_fileName, 'wb')
        # pickle full data set
        pickle.dump((tokenized_sentences, senti_Full_list), f, -1)
        # pickle tweetID
        pickle.dump(tweetID_Full_list, f, -1)
        f.close()       

    ###############################################################################
    return None


"""
####################################################################
"""












