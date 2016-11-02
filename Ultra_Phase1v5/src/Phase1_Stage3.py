

import json
import os
import numpy as np 
import pandas as pd
import collections as col

from RamSQL_TagUnique import RamSQL_Tag
from RamSQL_UserUnique import RamSQL_User


###################################################################################

def RamSQL_TagUnique_update(RamSQL_TagUnique, Tweet_OBJ, RollingScoreBank):

    # combine three tag sets together
    tag_all = set([])
    for item in Tweet_OBJ['Tag_Keyword']:
        tag_all.add(item)
    for item in Tweet_OBJ['Tag_Relevant']:
        tag_all.add(item)
    for item in Tweet_OBJ['Tag_due_user']:
        tag_all.add(item)

    # loop through all items of the tag set
    for tag in tag_all:
        
        # check if this tag already in RamSQL_TagUnique
        # if yes, update tweetID, user
        if tag in RamSQL_TagUnique:
            RamSQL_TagUnique[tag].update_N_score(tweetID=next(iter(Tweet_OBJ['tweet_id'])), user=next(iter(Tweet_OBJ['user_id'])))
        # if not, create entry
        elif tag not in RamSQL_TagUnique:
            RamSQL_TagUnique[tag] = RamSQL_Tag(tagText=tag, tweetID=next(iter(Tweet_OBJ['tweet_id'])), user=next(iter(Tweet_OBJ['user_id'])))       

        # update score for this tag
        score1 = 0.0
        Ncall1 = 0.0
        score2 = 0.0
        Ncall2 = 0.0
        # score and Ncall for keyword1
        if tag in RollingScoreBank['tag_keyword1']:
            score1 = 10.0
            Ncall1 = RollingScoreBank['tag_keyword1'][tag]
        elif tag in RollingScoreBank['tag_relevant1']:
            score1 = RollingScoreBank['tag_relevant1'][tag]
            Ncall1 = RollingScoreBank['tag_relevant1_N'][tag]
        else:
            score1 = 0.0
            Ncall1 = 0.0            
        # score and Ncall for keyword2
        if tag in RollingScoreBank['tag_keyword2']:
            score2 = 10.0
            Ncall2 = RollingScoreBank['tag_keyword2'][tag]
        elif tag in RollingScoreBank['tag_relevant2']:
            score2 = RollingScoreBank['tag_relevant2'][tag]
            Ncall2 = RollingScoreBank['tag_relevant2_N'][tag]
        else:
            score2 = 0.0
            Ncall2 = 0.0
        # update scores
        RamSQL_TagUnique[tag].update_score(score1=score1, Ncall1=Ncall1, score2=score2, Ncall2=Ncall2)

        # update other present tags
        for other_tags in tag_all:
            if other_tags != tag:
                RamSQL_TagUnique[tag].update_N_score(tagCon=other_tags)

    return RamSQL_TagUnique

###################################################################################

def RamSQL_UserUnique_update(RamSQL_UserUnique, Tweet_OBJ, RollingScoreBank):

    # combine three tag sets together
    tag_all = set([])
    for item in Tweet_OBJ['Tag_Keyword']:
        tag_all.add(item)
    for item in Tweet_OBJ['Tag_Relevant']:
        tag_all.add(item)
    for item in Tweet_OBJ['Tag_due_user']:
        tag_all.add(item)

    # TuserID
    Tuser = next(iter(Tweet_OBJ['user_id']))

    # if user already recorded
    if Tuser in RamSQL_UserUnique:
        RamSQL_UserUnique[Tuser].update_N_score(tweetID=next(iter(Tweet_OBJ['tweet_id'])), Ruser=next(iter(Tweet_OBJ['reply_to_userID'])))
    # if not, create entry
    elif Tuser not in RamSQL_UserUnique:
        RamSQL_UserUnique[Tuser] = RamSQL_User(tweetID=next(iter(Tweet_OBJ['tweet_id'])), Tuser=Tuser,TuserName=next(iter(Tweet_OBJ['user_name'])), 
            followers_count=next(iter(Tweet_OBJ['user_followers'])), friends_count=next(iter(Tweet_OBJ['user_friends'])))

    # Mentioned User Update
    for Muser_id in Tweet_OBJ['mentioned_userID']:
        RamSQL_UserUnique[Tuser].update_N_score(Muser=Muser_id)

    # Tags of this tweet
    for tags in tag_all:
        RamSQL_UserUnique[Tuser].update_N_score(tagCon=tags)

    # update scores of user
    RamSQL_UserUnique[Tuser].update_score(score1=RollingScoreBank['user1'][Tuser], Ncall1=RollingScoreBank['user1_N'][Tuser], 
        score2=RollingScoreBank['user2'][Tuser], Ncall2=RollingScoreBank['user2_N'][Tuser])

    return RamSQL_UserUnique

