import json
import os
import numpy as np 
import pandas as pd
import collections as col

########################################################################
# NOTE: this version of UTO_update enhanced the sensitivity of score to sudden upward change
# for BOTH tag and user
# For tag, threshold is 9.0 vs 2.0 and 5.0 vs 2.0, drop out 1.0
# For user, threshold is 9.0 vs 1.2 and 5.0 vs 1.2, drop out 0.1
# After threshold, need to set N_call = 1

def RollingScore_Update(RollingScoreBank, Tweet_OBJ, keyword1, keyword2,
                        thre_tag_mid, thre_tag_low, thre_tag_reset,
                        thre_user_mid, thre_user_low, thre_user_reset):

    #####################################################################################
    # 1st, if there is keyword_tag
    # update tag, then update user
    flag_key = False
    flag_key1 = False
    flag_key2 = False

    if len(Tweet_OBJ['Tag_Keyword']) > 0:
        flag_key = True

        ###############################################
        # add keyword_tag into RollingScoreBank
        for item in Tweet_OBJ['Tag_Keyword']:
            if keyword1 in item:
                RollingScoreBank['tag_keyword1'][item] += 1
                flag_key1 = True
            if keyword2 in item:
                RollingScoreBank['tag_keyword2'][item] += 1
                flag_key2 = True
        
        ###############################################
        # calculate cross scores for keyword tags
        for item in Tweet_OBJ['Tag_Keyword']:
            # tag of keyword2 for scores relative to keyword1
            if item in RollingScoreBank['tag_keyword2'] and flag_key1 == True:
                if item in RollingScoreBank['tag_relevant1']:
                    # update score; as New Average
                    New_Tag_Score = 1.0*(9.0 + RollingScoreBank['tag_relevant1'][item]*RollingScoreBank['tag_relevant1_N'][item])/(RollingScoreBank['tag_relevant1_N'][item]+1)
                    if RollingScoreBank['tag_relevant1'][item] < thre_tag_reset:
                        RollingScoreBank['tag_relevant1'][item] = 9.0
                        RollingScoreBank['tag_relevant1_N'][item] = 1
                    else:
                        RollingScoreBank['tag_relevant1'][item] = New_Tag_Score
                        RollingScoreBank['tag_relevant1_N'][item] += 1
                else:
                    RollingScoreBank['tag_relevant1'][item] = 9.0
                    RollingScoreBank['tag_relevant1_N'][item] = 1
            elif item in RollingScoreBank['tag_keyword2'] and flag_key1 == False:
                # calculate key2_rel_key1 score 
                Max_Score_key2_vkey1 = 0.0
                for tag_rel_1 in Tweet_OBJ['Tag_Relevant']:
                    if tag_rel_1 in RollingScoreBank['tag_relevant1'] and Max_Score_key2_vkey1 < RollingScoreBank['tag_relevant1'][tag_rel_1]:
                        Max_Score_key2_vkey1 = RollingScoreBank['tag_relevant1'][tag_rel_1]             
                # update accordingly
                if item in RollingScoreBank['tag_relevant1']:
                    # update score; as New Average
                    New_Tag_Score = 1.0*(Max_Score_key2_vkey1-1 + RollingScoreBank['tag_relevant1'][item]*RollingScoreBank['tag_relevant1_N'][item])/(RollingScoreBank['tag_relevant1_N'][item]+1)
                    if RollingScoreBank['tag_relevant1'][item] < thre_tag_reset:
                        RollingScoreBank['tag_relevant1'][item] = 9.0
                        RollingScoreBank['tag_relevant1_N'][item] = 1
                    else:
                        RollingScoreBank['tag_relevant1'][item] = New_Tag_Score
                        RollingScoreBank['tag_relevant1_N'][item] += 1
                else:
                    RollingScoreBank['tag_relevant1'][item] = 9.0
                    RollingScoreBank['tag_relevant1_N'][item] = 1

            ###############################################
            # tag of keyword1 for scores relative to keyword2
            if item in RollingScoreBank['tag_keyword1'] and flag_key2 == True:
                if item in RollingScoreBank['tag_relevant2']:
                    # update score; as New Average
                    New_Tag_Score = 1.0*(9.0 + RollingScoreBank['tag_relevant2'][item]*RollingScoreBank['tag_relevant2_N'][item])/(RollingScoreBank['tag_relevant2_N'][item]+1)
                    if RollingScoreBank['tag_relevant2'][item] < thre_tag_reset:
                        RollingScoreBank['tag_relevant2'][item] = 9.0
                        RollingScoreBank['tag_relevant2_N'][item] = 1
                    else:
                        RollingScoreBank['tag_relevant2'][item] = New_Tag_Score
                        RollingScoreBank['tag_relevant2_N'][item] += 1
                else:
                    RollingScoreBank['tag_relevant2'][item] = 9.0
                    RollingScoreBank['tag_relevant2_N'][item] = 1
            elif item in RollingScoreBank['tag_keyword1'] and flag_key2 == False:
                # calculate key2_rel_key1 score 
                Max_Score_key1_vkey2 = 0.0
                for tag_rel_2 in Tweet_OBJ['Tag_Relevant']:
                    if tag_rel_2 in RollingScoreBank['tag_relevant2'] and Max_Score_key1_vkey2 < RollingScoreBank['tag_relevant2'][tag_rel_2]:
                        Max_Score_key1_vkey2 = RollingScoreBank['tag_relevant2'][tag_rel_2]             
                # update accordingly
                if item in RollingScoreBank['tag_relevant2']:
                    # update score; as New Average
                    New_Tag_Score = 1.0*(Max_Score_key1_vkey2-1 + RollingScoreBank['tag_relevant2'][item]*RollingScoreBank['tag_relevant2_N'][item])/(RollingScoreBank['tag_relevant2_N'][item]+1)
                    if RollingScoreBank['tag_relevant2'][item] < thre_tag_reset:
                        RollingScoreBank['tag_relevant2'][item] = 9.0
                        RollingScoreBank['tag_relevant2_N'][item] = 1
                    else:
                        RollingScoreBank['tag_relevant2'][item] = New_Tag_Score
                        RollingScoreBank['tag_relevant2_N'][item] += 1
                else:
                    RollingScoreBank['tag_relevant2'][item] = 9.0
                    RollingScoreBank['tag_relevant2_N'][item] = 1

        ###############################################
        # add relvent_tag into RollingScoreBank; add to BOTH sides
        # since there is the keyword_tag, Max_score = 9
        for item in Tweet_OBJ['Tag_Relevant']:
            # keyword1
            if flag_key1:
                if item in RollingScoreBank['tag_relevant1']:
                    # update score; as New Average
                    New_Tag_Score = 1.0*(9.0 + RollingScoreBank['tag_relevant1'][item]*RollingScoreBank['tag_relevant1_N'][item])/(RollingScoreBank['tag_relevant1_N'][item]+1)
                    if RollingScoreBank['tag_relevant1'][item] < thre_tag_reset:
                        RollingScoreBank['tag_relevant1'][item] = 9.0
                        RollingScoreBank['tag_relevant1_N'][item] = 1
                    else:
                        RollingScoreBank['tag_relevant1'][item] = New_Tag_Score
                        RollingScoreBank['tag_relevant1_N'][item] += 1
                else:
                    RollingScoreBank['tag_relevant1'][item] = 9.0
                    RollingScoreBank['tag_relevant1_N'][item] = 1
            # keyword2
            if flag_key2:
                if item in RollingScoreBank['tag_relevant2']:
                    # update score; as New Average
                    New_Tag_Score = 1.0*(9.0 + RollingScoreBank['tag_relevant2'][item]*RollingScoreBank['tag_relevant2_N'][item])/(RollingScoreBank['tag_relevant2_N'][item]+1)
                    if RollingScoreBank['tag_relevant2'][item] < thre_tag_reset:
                        RollingScoreBank['tag_relevant2'][item] = 9.0
                        RollingScoreBank['tag_relevant2_N'][item] = 1
                    else:
                        RollingScoreBank['tag_relevant2'][item] = New_Tag_Score
                        RollingScoreBank['tag_relevant2_N'][item] += 1
                else:
                    RollingScoreBank['tag_relevant2'][item] = 9.0
                    RollingScoreBank['tag_relevant2_N'][item] = 1

        ###############################################
        # update user_infor
        for element in Tweet_OBJ['user_id']:
            user_id_str = element
        # keyword1
        if flag_key1: 
            if user_id_str in RollingScoreBank['user1'] and user_id_str in RollingScoreBank['user1_N']:
                # since there is the keyword_tag, score to 10
                New_User_Score = 1.0*(9.0 + RollingScoreBank['user1'][user_id_str]*RollingScoreBank['user1_N'][user_id_str])/(RollingScoreBank['user1_N'][user_id_str]+1)
                if RollingScoreBank['user1'][user_id_str] < thre_user_reset:
                    RollingScoreBank['user1'][user_id_str] = 9.0
                    RollingScoreBank['user1_N'][user_id_str] = 1
                else:
                    RollingScoreBank['user1'][user_id_str] = New_User_Score
                    RollingScoreBank['user1_N'][user_id_str] += 1
            else:
                RollingScoreBank['user1'][user_id_str] = 9.0
                RollingScoreBank['user1_N'][user_id_str] = 1            
        # keyword2
        if flag_key2: 
            if user_id_str in RollingScoreBank['user2'] and user_id_str in RollingScoreBank['user2_N']:
                # since there is the keyword_tag, score to 10
                New_User_Score = 1.0*(9.0 + RollingScoreBank['user2'][user_id_str]*RollingScoreBank['user2_N'][user_id_str])/(RollingScoreBank['user2_N'][user_id_str]+1)
                if RollingScoreBank['user2'][user_id_str] < thre_user_reset:
                    RollingScoreBank['user2'][user_id_str] = 9.0
                    RollingScoreBank['user2_N'][user_id_str] = 1
                else:
                    RollingScoreBank['user2'][user_id_str] = New_User_Score
                    RollingScoreBank['user2_N'][user_id_str] += 1
            else:
                RollingScoreBank['user2'][user_id_str] = 9.0
                RollingScoreBank['user2_N'][user_id_str] = 1

    #####################################################################################

    #####################################################################################
    # 2nd, if not, whether there is upper_tag
    flag_relevent = False
    if len(Tweet_OBJ['Tag_Relevant']) > 0: 
        flag_relevent = True

    # keyword1
    # add relvent_tag into RollingScoreBank
    # find out max_score
    # include cases: for keyword1 side, no tag_keyword1 or no tag_keyword both
    if flag_key1 == False and flag_relevent == True:
        Max_Score_1 = 0.0
        for item in Tweet_OBJ['Tag_Relevant']:
            if item in RollingScoreBank['tag_relevant1'] and Max_Score_1 < RollingScoreBank['tag_relevant1'][item]:
                Max_Score_1 = RollingScoreBank['tag_relevant1'][item]

        # load relevant_tags
        for item in Tweet_OBJ['Tag_Relevant']:
            if item in RollingScoreBank['tag_relevant1']:
                # update score; as New Average
                New_Tag_Score = 1.0*(Max_Score_1 - 1 + RollingScoreBank['tag_relevant1'][item]*RollingScoreBank['tag_relevant1_N'][item])/(RollingScoreBank['tag_relevant1_N'][item]+1)
                if Max_Score_1 > thre_tag_mid and RollingScoreBank['tag_relevant1'][item] < thre_tag_reset:
                    RollingScoreBank['tag_relevant1'][item] = Max_Score_1 -1
                    RollingScoreBank['tag_relevant1_N'][item] = 1
                else:
                    RollingScoreBank['tag_relevant1'][item] = New_Tag_Score
                    RollingScoreBank['tag_relevant1_N'][item] += 1
            else:
                RollingScoreBank['tag_relevant1'][item] = Max_Score_1 - 1
                RollingScoreBank['tag_relevant1_N'][item] = 1

    # keyword2
    # add relvent_tag into RollingScoreBank
    # find out max_score
    if flag_key2 == False and flag_relevent == True:
        Max_Score_2 = 0.0
        for item in Tweet_OBJ['Tag_Relevant']:
            if item in RollingScoreBank['tag_relevant2'] and Max_Score_2 < RollingScoreBank['tag_relevant2'][item]:
                Max_Score_2 = RollingScoreBank['tag_relevant2'][item]

        # load relevant_tags
        for item in Tweet_OBJ['Tag_Relevant']:
            if item in RollingScoreBank['tag_relevant2']:
                # update score; as New Average
                New_Tag_Score = 1.0*(Max_Score_2 - 1 + RollingScoreBank['tag_relevant2'][item]*RollingScoreBank['tag_relevant2_N'][item])/(RollingScoreBank['tag_relevant2_N'][item]+1)
                if Max_Score_2 > thre_tag_mid and RollingScoreBank['tag_relevant2'][item] < thre_tag_reset:
                    RollingScoreBank['tag_relevant2'][item] = Max_Score_2 -1
                    RollingScoreBank['tag_relevant2_N'][item] = 1
                else:
                    RollingScoreBank['tag_relevant2'][item] = New_Tag_Score
                    RollingScoreBank['tag_relevant2_N'][item] += 1
            else:
                RollingScoreBank['tag_relevant2'][item] = Max_Score_2 - 1
                RollingScoreBank['tag_relevant2_N'][item] = 1

    # update user_infor
    for element in Tweet_OBJ['user_id']:
        user_id_str = element
    # keyword1  
    if flag_key1 == False and flag_relevent == True:
        if user_id_str in RollingScoreBank['user1'] and user_id_str in RollingScoreBank['user1_N']:
            # compare user_score with current Max_Score_1
            New_User_Score = 1.0*(Max_Score_1 - 1 + RollingScoreBank['user1'][user_id_str]*RollingScoreBank['user1_N'][user_id_str])/(RollingScoreBank['user1_N'][user_id_str]+1)
            if  RollingScoreBank['user1'][user_id_str] < thre_user_reset and Max_Score_1 > thre_user_mid:
                RollingScoreBank['user1'][user_id_str] = Max_Score_1 -1
                RollingScoreBank['user1_N'][user_id_str] = 1
            else:
                RollingScoreBank['user1'][user_id_str] = New_User_Score
                RollingScoreBank['user1_N'][user_id_str] += 1
        else:
            RollingScoreBank['user1'][user_id_str] = Max_Score_1 - 1
            RollingScoreBank['user1_N'][user_id_str] = 1
    # keyword2  
    if flag_key2 == False and flag_relevent == True:
        if user_id_str in RollingScoreBank['user2'] and user_id_str in RollingScoreBank['user2_N']:
            # compare user_score with current Max_Score_2
            New_User_Score = 1.0*(Max_Score_2 - 1 + RollingScoreBank['user2'][user_id_str]*RollingScoreBank['user2_N'][user_id_str])/(RollingScoreBank['user2_N'][user_id_str]+1)
            if  RollingScoreBank['user2'][user_id_str] < thre_user_reset and Max_Score_2 > thre_user_mid:
                RollingScoreBank['user2'][user_id_str] = Max_Score_2 -1
                RollingScoreBank['user2_N'][user_id_str] = 1
            else:
                RollingScoreBank['user2'][user_id_str] = New_User_Score
                RollingScoreBank['user2_N'][user_id_str] += 1
        else:
            RollingScoreBank['user2'][user_id_str] = Max_Score_2 - 1
            RollingScoreBank['user2_N'][user_id_str] = 1

    ########################################################################

    ########################################################################
    # 3rd, if no keyword nor relevant tags, update RollingScoreBank according to Tag_due_user
    # Thus, upload ALL remaining tags into the Bank
    if flag_relevent == False and flag_key == False and len(Tweet_OBJ['Tag_due_user']) > 0:
        # add Tag_due_user into RollingScoreBank
        # find out max_score

        # keyword1
        Max_Score_1 = 0.0
        for item in Tweet_OBJ['Tag_due_user']:
            if item in RollingScoreBank['tag_relevant1'] and Max_Score_1 < RollingScoreBank['tag_relevant1'][item]:
                Max_Score_1 = RollingScoreBank['tag_relevant1'][item]
        # load relevant_tags
        for item in Tweet_OBJ['Tag_due_user']:
            if item in RollingScoreBank['tag_relevant1']:
                # update score; as New Average
                New_Tag_Score = 1.0*(Max_Score_1 - 1 + RollingScoreBank['tag_relevant1'][item]*RollingScoreBank['tag_relevant1_N'][item])/(RollingScoreBank['tag_relevant1_N'][item]+1)
                RollingScoreBank['tag_relevant1'][item] = New_Tag_Score
                # update N_call
                RollingScoreBank['tag_relevant1_N'][item] += 1
            else:
                RollingScoreBank['tag_relevant1'][item] = Max_Score_1 - 1
                RollingScoreBank['tag_relevant1_N'][item] = 1

        # keyword2
        Max_Score_2 = 0.0
        for item in Tweet_OBJ['Tag_due_user']:
            if item in RollingScoreBank['tag_relevant2'] and Max_Score_2 < RollingScoreBank['tag_relevant2'][item]:
                Max_Score_2 = RollingScoreBank['tag_relevant2'][item]
        # load relevant_tags
        for item in Tweet_OBJ['Tag_due_user']:
            if item in RollingScoreBank['tag_relevant2']:
                # update score; as New Average
                New_Tag_Score = 1.0*(Max_Score_2 - 1 + RollingScoreBank['tag_relevant2'][item]*RollingScoreBank['tag_relevant2_N'][item])/(RollingScoreBank['tag_relevant2_N'][item]+1)
                RollingScoreBank['tag_relevant2'][item] = New_Tag_Score
                # update N_call
                RollingScoreBank['tag_relevant2_N'][item] += 1
            else:
                RollingScoreBank['tag_relevant2'][item] = Max_Score_2 - 1
                RollingScoreBank['tag_relevant2_N'][item] = 1

        # update user_infor
        for element in Tweet_OBJ['user_id']:
            user_id_str = element
        # keyword1
        if user_id_str in RollingScoreBank['user1'] and user_id_str in RollingScoreBank['user1_N']:
            # compare user_score with current Max_Score_1
            RollingScoreBank['user1_N'][user_id_str] += 1
            New_User_Score = 1.0*(Max_Score_1 - 1 + RollingScoreBank['user1'][user_id_str]*RollingScoreBank['user1_N'][user_id_str])/(RollingScoreBank['user1_N'][user_id_str]+1)       
            RollingScoreBank['user1'][user_id_str] = New_User_Score
        else:
            RollingScoreBank['user1'][user_id_str] = Max_Score_1 - 1
            RollingScoreBank['user1_N'][user_id_str] = 1
        # keyword2
        if user_id_str in RollingScoreBank['user2'] and user_id_str in RollingScoreBank['user2_N']:
            # compare user_score with current Max_Score_1
            RollingScoreBank['user2_N'][user_id_str] += 1
            New_User_Score = 1.0*(Max_Score_1 - 1 + RollingScoreBank['user2'][user_id_str]*RollingScoreBank['user2_N'][user_id_str])/(RollingScoreBank['user2_N'][user_id_str]+1)       
            RollingScoreBank['user2'][user_id_str] = New_User_Score
        else:
            RollingScoreBank['user2'][user_id_str] = Max_Score_1 - 1
            RollingScoreBank['user2_N'][user_id_str] = 1

    ########################################################################

    ########################################################################
    # clean the bank according to ~25% threshold of ~score >= 1
    
    # clear tags for keyword1
    clear_tag = set([])
    # search for tags
    for key in RollingScoreBank['tag_relevant1']:
        if RollingScoreBank['tag_relevant1'][key] < thre_tag_low:
            clear_tag.add(key)
            #print "delete tag: ", key, RollingScoreBank['tag_relevant'][key]
    # clear targets
    for key in clear_tag:
        del RollingScoreBank['tag_relevant1'][key]
        del RollingScoreBank['tag_relevant1_N'][key]

    # clear tags for keyword2
    clear_tag = set([])
    # search for tags
    for key in RollingScoreBank['tag_relevant2']:
        if RollingScoreBank['tag_relevant2'][key] < thre_tag_low:
            clear_tag.add(key)
            #print "delete tag: ", key, RollingScoreBank['tag_relevant'][key]
    # clear targets
    for key in clear_tag:
        del RollingScoreBank['tag_relevant2'][key]
        del RollingScoreBank['tag_relevant2_N'][key]

    ########################################################################

    ########################################################################
    
    # clear users for keyword1
    clear_user = set([])
    # search for tags
    for key in RollingScoreBank['user1']:
        if RollingScoreBank['user1'][key] < thre_user_low:
            clear_user.add(key)
            #print "delete user: ", key, RollingScoreBank['user'][key]
    # clear targets
    for key in clear_user:
        del RollingScoreBank['user1'][key]
        del RollingScoreBank['user1_N'][key]

    # clear users for keyword2
    clear_user = set([])
    # search for tags
    for key in RollingScoreBank['user2']:
        if RollingScoreBank['user2'][key] < thre_user_low:
            clear_user.add(key)
            #print "delete user: ", key, RollingScoreBank['user'][key]
    # clear targets
    for key in clear_user:
        del RollingScoreBank['user2'][key]
        del RollingScoreBank['user2_N'][key]

    ########################################################################
    # return

    return RollingScoreBank



