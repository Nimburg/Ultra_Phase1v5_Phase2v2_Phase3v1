
import json
import os
import numpy as np 
import statistics
import pandas as pd
import collections as col
import csv

import pymysql.cursors


'''
####################################################################
'''

def MarkedTag_Import(file_name_list):
    '''
    file_name_list: list of .csv files; 2 files, 1st for keyword 1 as trump
    '''
    # dict(), as main results structure
    # key as 'tags', value is tuple (int, int)
    # there is no separate dict() for keyword 1 & 2, as some tags over laps
    MarkedTags_dict = dict()

    counter_keyword = 0
    for file_name in file_name_list: 
        print "reading file: ", file_name
        counter_keyword += 1
        
        # read .csv file
        InputfileDir = os.path.dirname(os.path.realpath('__file__'))
        data_file_name =  '../Data/' + file_name
        Inputfilename = os.path.join(InputfileDir, data_file_name) # ../ get back to upper level
        Inputfilename = os.path.abspath(os.path.realpath(Inputfilename))
        print Inputfilename
        file_input = open(Inputfilename,'r')
        csv_data = csv.reader(file_input, delimiter=',')
        next(csv_data, None) # skip header

        # go through lines
        for row in csv_data:
            # tag, string format, ascii
            tag_text = str( row[0] ).decode('utf-8').encode('ascii', 'ignore')
            # check is tag already in MarkedTags_dict
            # initialize as (0,0)
            if tag_text not in MarkedTags_dict:
                MarkedTags_dict[tag_text] = ( 0.0, 0.0)
            # insert value
            # keyword 1
            if counter_keyword == 1:
                MarkedTags_dict[tag_text] = tuple( [ sum(x) for x in zip( MarkedTags_dict[tag_text],
                                                                        ( float(row[1]), 0.0 ) 
                                                                        ) ] 
                                                 )
                # print tag_text, MarkedTags_dict[tag_text]
            # keyword 2
            if counter_keyword == 2:
                MarkedTags_dict[tag_text] = tuple( [ sum(x) for x in zip( MarkedTags_dict[tag_text],
                                                                        ( 0.0 , float(row[1]) ) 
                                                                        ) ] 
                                                 )
                # print tag_text, MarkedTags_dict[tag_text]
    # return dict()
    return MarkedTags_dict


'''
####################################################################
'''

def Iteration_TagSentiScores(MarkedTags_dict, Tags_Today_preIter, Tags_PreviousDay_postIter, 
                             max_IterPeriod=100, 
                             thre_senti_score1=0.01, thre_senti_score2=0.01,
                             thre_relev_score1=1.0, thre_relev_score2=1.0, 
                             thre_Ncall1=100, thre_Ncall2=100, 
                             thre_dayCall=10):
    '''
    MarkedTags_dict: dict() of tags with tuple scores of (score_keyword1, score_keyword2)
                     constant, scores marked by human

    Tags_Today_preIter: col.defaultdict(col.defaultdict) for present day's tags
    Tags_PreviousDay_postIter: of previous day

    present_date: pd.to_datetime, of present date

    max_IterPeriod: max number of iteration cycles
    thre_senti_score1: thresholds to determine if senti_scores have converged through iteration
    thre_senti_score2: 

    thre_relev_score1: threshold on relevence score
    thre_relev_score2: 
    thre_Ncall1: threshold on cumulated calls
    thre_Ncall2: 
    thre_dayCall: threshold on one-day Ncalls

    jsonOutput_fileNameHeader: file name of .json save of Tags_Today_postIter
    
    --------------------------------
    Tags_Today_postIter need to have:
    # sentiment scores
    Tags_Today_postIter['tag']['senti_score1']
    Tags_Today_postIter['tag']['senti_score2']
    # statistics on relevence scores
    Tags_Today_postIter['tag']['score1_ave']
    Tags_Today_postIter['tag']['score1_std']
    Tags_Today_postIter['tag']['score1_median'] <- statistics.median(list)
    Tags_Today_postIter['tag']['score1_max'] <- max, min plus ave, median provides an estimation of stability
    Tags_Today_postIter['tag']['score1_min'] <- whether ave is close to the stable value or still decaying
    # statistics on relevence calls
    # currently unnecessary
    # statistics on degree_tags
    Tags_Today_postIter['tag']['tag_degree']

    --------------------------------
    calculation of senti_score

    if tag in MarkedTags_dict: do NOT change
    
    0th iteration
    New_senti_score = previous_senti_score if exists

    Nth iteration
    New_senti_score = ('New_senti_score' + Networked_tags_score)/2

    Networked_tags_score = 1/presentDay_total_Ntags * ( N_eachTag_Call * score_eachTag_presentDay )

    Stop either when max_IterPeriod reached, 
    Or when changes in Networked_tags_score are small enough
    '''

    #################################
    # filters on Tags_Today_preIter # 
    #################################

    print "Number of tags in Tags_Today_preIter before filtering: %i" % len(Tags_Today_preIter)

    # going through each tag in Tags_Today_preIter
    tag_list = []
    for tag in Tags_Today_preIter:
        tag_list.append(tag)
    # go through tag_list
    for tag in tag_list:

        if Tags_Today_preIter[tag]['score1_fin'] < thre_relev_score1:
            del Tags_Today_preIter[tag]
            continue # done filtering for tag

        if Tags_Today_preIter[tag]['score2_fin'] < thre_relev_score2:
            del Tags_Today_preIter[tag]
            continue # done filtering for tag

        if Tags_Today_preIter[tag]['Ncall1_fin'] < thre_Ncall1:
            del Tags_Today_preIter[tag]
            continue # done filtering for tag

        if Tags_Today_preIter[tag]['Ncall2_fin'] < thre_Ncall2:
            del Tags_Today_preIter[tag]
            continue # done filtering for tag
        if Tags_Today_preIter[tag]['totalCall'] < thre_dayCall:
            del Tags_Today_preIter[tag]
            continue # done filtering for tag

    print "Number of tags in Tags_Today_preIter after filtering: %i" % len(Tags_Today_preIter)

    ####################################################################

    ##################################################
    # initialize new variables in Tags_Today_preIter # 
    ##################################################
    # going through each tag in Tags_Today_preIter
    for tag in Tags_Today_preIter:
        # for statistics on degree_tags
        Tags_Today_preIter[tag]['tag_degree'] = len( Tags_Today_preIter[tag]['tag_counter'] )

        # statistics on relevence scores
        if len( Tags_Today_preIter[tag]['tagScore1_list_float'] ) > 0:
            Tags_Today_preIter[tag]['score1_ave'] = statistics.mean( Tags_Today_preIter[tag]['tagScore1_list_float'] )
            Tags_Today_preIter[tag]['score1_std'] = statistics.stdev( Tags_Today_preIter[tag]['tagScore1_list_float'] )
            Tags_Today_preIter[tag]['score1_median'] = statistics.median( Tags_Today_preIter[tag]['tagScore1_list_float'] )
            Tags_Today_preIter[tag]['score1_max'] = max( Tags_Today_preIter[tag]['tagScore1_list_float'] )
            Tags_Today_preIter[tag]['score1_min'] = min( Tags_Today_preIter[tag]['tagScore1_list_float'] )
        else:
            Tags_Today_preIter[tag]['score1_ave'] = -1
            Tags_Today_preIter[tag]['score1_std'] = -1
            Tags_Today_preIter[tag]['score1_median'] = -1
            Tags_Today_preIter[tag]['score1_max'] = -1
            Tags_Today_preIter[tag]['score1_min'] = -1      

        if len( Tags_Today_preIter[tag]['tagScore2_list_float'] ) > 0:
            Tags_Today_preIter[tag]['score2_ave'] = statistics.mean( Tags_Today_preIter[tag]['tagScore2_list_float'] )
            Tags_Today_preIter[tag]['score2_std'] = statistics.stdev( Tags_Today_preIter[tag]['tagScore2_list_float'] )
            Tags_Today_preIter[tag]['score2_median'] = statistics.median( Tags_Today_preIter[tag]['tagScore2_list_float'] )
            Tags_Today_preIter[tag]['score2_max'] = max( Tags_Today_preIter[tag]['tagScore2_list_float'] )
            Tags_Today_preIter[tag]['score2_min'] = min( Tags_Today_preIter[tag]['tagScore2_list_float'] )
        else:
            Tags_Today_preIter[tag]['score2_ave'] = -1
            Tags_Today_preIter[tag]['score2_std'] = -1
            Tags_Today_preIter[tag]['score2_median'] = -1
            Tags_Today_preIter[tag]['score2_max'] = -1
            Tags_Today_preIter[tag]['score2_min'] = -1  

        # Initialize senti_score1&2
        # if tag marked by human
        if tag in MarkedTags_dict: 
            Tags_Today_preIter[tag]['senti_score1'] = MarkedTags_dict[tag][0]
            Tags_Today_preIter[tag]['senti_score2'] = MarkedTags_dict[tag][1]
        # if tag not marked by human, but appeared before
        if (tag not in MarkedTags_dict) and (tag in Tags_PreviousDay_postIter):
            Tags_Today_preIter[tag]['senti_score1'] = Tags_PreviousDay_postIter[tag]['senti_score1']
            Tags_Today_preIter[tag]['senti_score2'] = Tags_PreviousDay_postIter[tag]['senti_score2']        
        # if tag is completely new
        if (tag not in MarkedTags_dict) and (tag not in Tags_PreviousDay_postIter):
            # neutral setting
            Tags_Today_preIter[tag]['senti_score1'] = 0.0
            Tags_Today_preIter[tag]['senti_score2'] = 0.0

    ####################################################################

    ####################################################
    # Iteration on Tags_Today_preIter for senti_scores # 
    ####################################################

    counter_Iter = 0
    flag_Iter_Stable = False
    
    while (counter_Iter <= max_IterPeriod) and (flag_Iter_Stable == False):

        counter_Iter += 1

        flag_Iter_Stable = True  
        counter_Iter_unstable1 = 0
        counter_Iter_unstable2 = 0

        # going through each tag in Tags_Today_preIter
        for tag_iter in Tags_Today_preIter:
            # if tag marked by human
            if tag_iter in MarkedTags_dict:
                continue # pass opeartions for this tag
            # if this tag has to networked_tags
            if Tags_Today_preIter[tag_iter]['tag_degree'] == 0: 
                continue # pass opeartions for this tag

            ###############################################
            # calculate perIter_senti_score1 for this tag #
            ###############################################
            
            perIter_senti_score1 = 0.0
            perIter_senti_score2 = 0.0
            perTag_total_NetCall = 0 # total number of networked tags appearance
            for tag_networked in Tags_Today_preIter[tag_iter]['tag_counter']: 
                
                # if tag_networked is hand-marked
                if tag_networked in MarkedTags_dict:            
                    perIter_senti_score1 += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked] * \
                                            MarkedTags_dict[tag_networked][0]
                    perIter_senti_score2 += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked] * \
                                            MarkedTags_dict[tag_networked][1]               
                    perTag_total_NetCall += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked]
                    continue # done updating scores from tag_networked

                # if tag is not hand-marked, but showed up today
                if (tag_networked not in MarkedTags_dict) and (tag_networked in Tags_Today_preIter):
                    perIter_senti_score1 += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked] * \
                                            Tags_Today_preIter[tag_networked]['senti_score1']
                    perIter_senti_score2 += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked] * \
                                            Tags_Today_preIter[tag_networked]['senti_score2']               
                    perTag_total_NetCall += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked]
                    continue # done updating scores from tag_networked

                # if tag only showed up in the previous day
                if (tag_networked not in MarkedTags_dict) and (tag_networked not in Tags_Today_preIter) and (tag_networked in Tags_PreviousDay_postIter):
                    perIter_senti_score1 += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked] * \
                                            Tags_PreviousDay_postIter[tag_networked]['senti_score1']
                    perIter_senti_score2 += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked] * \
                                            Tags_PreviousDay_postIter[tag_networked]['senti_score2']                
                    perTag_total_NetCall += Tags_Today_preIter[tag_iter]['tag_counter'][tag_networked]
                    continue # done updating scores from tag_networked

            # end of for tag_networked in Tags_Today_preIter[tag_iter]['tag_counter']
            if perTag_total_NetCall > 0:
                perIter_senti_score1 = 1.0* perIter_senti_score1 / perTag_total_NetCall
                perIter_senti_score2 = 1.0* perIter_senti_score2 / perTag_total_NetCall
            else:
                # reset if sth is wrong with perTag_total_NetCall
                perIter_senti_score1 = 0.0
                perIter_senti_score2 = 0.0

            # compare perIter_senti_score1 with present_day's senti_score, for updating and flagging
            # sneti_score is between 0 and 1; 
            if abs( perIter_senti_score1 - Tags_Today_preIter[tag_iter]['senti_score1']) >= thre_senti_score1:
                flag_Iter_Stable = False
                counter_Iter_unstable1 += 1
                # early stop for converging scores
                Tags_Today_preIter[tag_iter]['senti_score1'] = (Tags_Today_preIter[tag_iter]['senti_score1'] + perIter_senti_score1)/2.0

            # compare perIter_senti_score2 with present_day's senti_score, for updating and flagging
            # sneti_score is between 0 and 1; 
            if abs( perIter_senti_score2 - Tags_Today_preIter[tag_iter]['senti_score2']) >= thre_senti_score2:
                flag_Iter_Stable = False
                counter_Iter_unstable2 += 1
                # early stop for converging scores
                Tags_Today_preIter[tag_iter]['senti_score2'] = (Tags_Today_preIter[tag_iter]['senti_score2'] + perIter_senti_score2)/2.0

        # end of for tag_iter in Tags_Today_preIter:
    
        print "Finished Iteration: %i" % counter_Iter
        print "%i out of %i unstable for senti_score1" % tuple([counter_Iter_unstable1]+[len(Tags_Today_preIter)])
        print "%i out of %i unstable for senti_score2" % tuple([counter_Iter_unstable2]+[len(Tags_Today_preIter)])
    # end of while :
    
    ####################################################################
    # effectively as Tags_Today_postIter
    return Tags_Today_preIter


'''
####################################################################
'''



