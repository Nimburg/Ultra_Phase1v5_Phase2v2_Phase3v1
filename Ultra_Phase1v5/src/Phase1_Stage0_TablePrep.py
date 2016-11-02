

import json
import os
import numpy as np 
import pandas as pd
import collections as col

import pymysql.cursors

####################################################################
# variable type check
def check_args(*types):
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            for val, typ in zip(args, types):
                assert isinstance(val, typ), "Value {} is not of expected type {}".format(val, typ)
            return func(*args, **kwargs)
        return wrapper
    return real_decorator

####################################################################

# single input pd.timestamp
@check_args(pd.tslib.Timestamp)
def pd_timestamp_check(input):
    return True

"""
####################################################################
"""

def TweetStack_Init(connection):
    
    #Comd
    # Do NOT drop Tweet_Stack; Luigid consideration
    Comd_TweetStack_Init = """
CREATE TABLE IF NOT EXISTS Tweet_Stack
(
    tweetID BIGINT PRIMARY KEY NOT NULL,
    tweetTime TIMESTAMP NOT NULL,
    userID BIGINT NOT NULL,
    tweetText TEXT COLLATE utf8_bin,
    reply_user_id BIGINT,
    MenUserList_text TEXT,
    TagList_text TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin"""
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute(Comd_TweetStack_Init)
        # commit commands
        print "Tweet_Stack Initialized"
        connection.commit()
    finally:
        pass

####################################################################

def TweetStack_load(connection, Tweet_OBJ):

    # DataTypeCheck for pd.timestamp; compare and update pin_time
    flag_timeStamp = False  
    try:
        flag_timeStamp = pd_timestamp_check(next(iter(Tweet_OBJ['tweet_time'])))
    except AssertionError:
        print "pin_time datatype failed"
        pass 
    else:
        pass
    
    # id_str check
    flag_id_str = False
    if next(iter(Tweet_OBJ['tweet_id'])).isdigit() and next(iter(Tweet_OBJ['user_id'])).isdigit() and next(iter(Tweet_OBJ['reply_to_userID'])).isdigit():
        flag_id_str = True
    
    # mentioned_userID check and parse
    mentioned_userID_str = ""
    for men_user in Tweet_OBJ['mentioned_userID']:
        if men_user.isdigit():
            mentioned_userID_str += men_user + ','
    if len(mentioned_userID_str) > 1:
        mentioned_userID_str = mentioned_userID_str[:-1] # for the last ','
        
    # parse tagList
    tagList_str = ""
    for tag_key in Tweet_OBJ['Tag_Keyword']:
        tagList_str += tag_key + ','
    for tag_rel in Tweet_OBJ['Tag_Relevant']:
        tagList_str += tag_rel + ','
    for tag_user in Tweet_OBJ['Tag_due_user']:
        tagList_str += tag_user + ','
    if len(tagList_str) > 1:
        tagList_str = tagList_str[:-1]

    ####################################################################
    if flag_timeStamp and flag_id_str:
        # command for Tweet_Stack
        comd_TweetStack_Insert = """
INSERT INTO Tweet_Stack (tweetID, tweetTime, userID, reply_user_id, tweetText, MenUserList_text, TagList_text)
VALUES ( %s, '%s', %s, %s, '%s', '%s', '%s')
ON DUPLICATE KEY UPDATE userID = %s;"""
        # execute commands
        user_id = next(iter(Tweet_OBJ['user_id']))
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_TweetStack_Insert % tuple( [next(iter(Tweet_OBJ['tweet_id']))] + 
                                                                [str(next(iter(Tweet_OBJ['tweet_time'])))] + 
                                                                [user_id] + 
                                                                [next(iter(Tweet_OBJ['reply_to_userID']))] + 
                                                                [next(iter(Tweet_OBJ['text']))] + 
                                                                [mentioned_userID_str] + 
                                                                [tagList_str] + 
                                                                [user_id]
                                                               )
                              )
            # commit commands 
            connection.commit()
        finally:
            pass


"""
####################################################################
"""

def TagUnique_Init(connection, pin_time):

    # table Name
    tableName = "TagUnique_"+pin_time.strftime('%Y_%m_%d_%H')
    #Comd
    Comd_TagUnique_Init = """
CREATE TABLE IF NOT EXISTS %s
(
    tagText varchar(255) PRIMARY KEY NOT NULL,
    totalCall int NOT NULL,
    score1_fin float,
    Ncall1_fin int,
    score2_fin float,
    Ncall2_fin int,
    tagScore1_text varchar(3000),
    tagNcall1_text varchar(3000),
    tagScore2_text varchar(3000),
    tagNcall2_text varchar(3000),
    user_counter_text varchar(3000),
    tag_counter_text varchar(3000)
)ENGINE=MEMORY;"""
    print Comd_TagUnique_Init
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_TagUnique_Init % tableName )
        # commit commands
        print tableName+" Initialized"
        connection.commit()
    finally:
        pass

####################################################################

def TagUnique_Insert(connection, pin_time, RamSQL_TagUnique):

    # table Name
    tableName = "TagUnique_"+pin_time.strftime('%Y_%m_%d_%H')
    
    # go throught each key, parse contents, and upload
    for key in RamSQL_TagUnique:
        
        # parse variables into strings, using class function
        tagScore1_text = RamSQL_TagUnique[key].tagScore1_str()
        if len(tagScore1_text) > 3000:
            tagScore1_text = tagScore1_text[:3000]
        
        tagNcall1_text = RamSQL_TagUnique[key].tagNcall1_str()
        if len(tagNcall1_text) > 3000:
            tagNcall1_text = tagNcall1_text[:3000]

        tagScore2_text = RamSQL_TagUnique[key].tagScore2_str()
        if len(tagScore2_text) > 3000:
            tagScore2_text = tagScore2_text[:3000]
        
        tagNcall2_text = RamSQL_TagUnique[key].tagNcall2_str()
        if len(tagNcall2_text) > 3000:
            tagNcall2_text = tagNcall2_text[:3000]

        user_counter_text = RamSQL_TagUnique[key].user_counter_str()
        if len(user_counter_text) > 3000:
            user_counter_text = user_counter_text[:3000]        
        
        tag_counter_text = RamSQL_TagUnique[key].tag_counter_str()
        if len(tag_counter_text) > 3000:
            tag_counter_text = tag_counter_text[:3000]

        # get score_fin and Ncall_fin
        score1_fin = 0
        if len(RamSQL_TagUnique[key].tagScore1) > 0:
            score1_fin = RamSQL_TagUnique[key].tagScore1[-1]

        Ncall1_fin = 0
        if len(RamSQL_TagUnique[key].tagNcall1) > 0:    
            Ncall1_fin = RamSQL_TagUnique[key].tagNcall1[-1]

        score2_fin = 0
        if len(RamSQL_TagUnique[key].tagScore2) > 0:
            score2_fin = RamSQL_TagUnique[key].tagScore2[-1]

        Ncall2_fin = 0
        if len(RamSQL_TagUnique[key].tagNcall2) > 0:
            Ncall2_fin = RamSQL_TagUnique[key].tagNcall2[-1]

        # check totalCall
        if RamSQL_TagUnique[key].totalCall != len(RamSQL_TagUnique[key].tagScore1) or RamSQL_TagUnique[key].totalCall != len(RamSQL_TagUnique[key].tweetID_list):
            print "Bad Total Call"

        # command for Tweet_Stack
        comd_TagUnique_Insert = """
INSERT INTO %s (tagText, totalCall, score1_fin, Ncall1_fin, score2_fin, Ncall2_fin, \
tagScore1_text, tagNcall1_text, tagScore2_text, tagNcall2_text, user_counter_text, tag_counter_text)
VALUES ( '%s', %s, %s, %s, %s, %s, '%s', '%s', '%s', '%s', '%s', '%s')
ON DUPLICATE KEY UPDATE totalCall = %s;"""
        # execute commands
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_TagUnique_Insert % tuple( [tableName] + 
                                                               [RamSQL_TagUnique[key].tagText] + 
                                                               [str(RamSQL_TagUnique[key].totalCall)] + 
                                                               [str(score1_fin)] + 
                                                               [str(Ncall1_fin)] + 
                                                               [str(score2_fin)] + 
                                                               [str(Ncall2_fin)] + 
                                                               [tagScore1_text] + 
                                                               [tagNcall1_text] + 
                                                               [tagScore2_text] + 
                                                               [tagNcall2_text] + 
                                                               [user_counter_text] + 
                                                               [tag_counter_text] + 
                                                               [str(RamSQL_TagUnique[key].totalCall)]
                                                              ) 
                              )
            # commit commands
            connection.commit()
        finally:
            pass


####################################################################

def TagUnique_Convert(connection, pin_time):

    # table Name
    tableName = "TagUnique_"+pin_time.strftime('%Y_%m_%d_%H')
    #Comd
    comd_convert = """
ALTER TABLE %s ENGINE=InnoDB;"""
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( comd_convert % tableName )
        # commit commands
        print tableName+" Converted"
        connection.commit()
    finally:
        pass

"""
####################################################################
"""

def UserUnique_Init(connection, pin_time):

    # table Name
    tableName = "UserUnique_"+pin_time.strftime('%Y_%m_%d_%H')
    #Comd
    Comd_UserUnique_Init = """
CREATE TABLE IF NOT EXISTS %s
(
    userID bigint PRIMARY KEY NOT NULL,
    userName varchar(255),
    totalAction int,
    followers int,
    friends int,
    score1_fin float,
    Ncall1_fin int,
    score2_fin float,
    Ncall2_fin int,
    userScore1_text varchar(3000),
    userNcall1_text varchar(3000),
    userScore2_text varchar(3000),
    userNcall2_text varchar(3000),
    Ruser_counter_text varchar(3000),
    Muser_counter_text varchar(3000),
    Tag_counter_text varchar(3000)
)ENGINE=MEMORY;"""
    print Comd_UserUnique_Init
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( Comd_UserUnique_Init % tableName )
        # commit commands
        print tableName+" Initialized"
        connection.commit()
    finally:
        pass

####################################################################

def UserUnique_Insert(connection, pin_time, RamSQL_UserUnique):

    # table Name
    tableName = "UserUnique_"+pin_time.strftime('%Y_%m_%d_%H')

    # go throught each key, parse contents, and upload
    for key in RamSQL_UserUnique:

        # parse variables into strings, using class function
        userScore1_text = RamSQL_UserUnique[key].userScore1_str()
        if len(userScore1_text) > 3000:
            userScore1_text = userScore1_text[:3000]
        
        userNcall1_text = RamSQL_UserUnique[key].userNcall1_str()
        if len(userNcall1_text) > 3000:
            userNcall1_text = userNcall1_text[:3000]

        userScore2_text = RamSQL_UserUnique[key].userScore2_str()
        if len(userScore2_text) > 3000:
            userScore2_text = userScore2_text[:3000]

        userNcall2_text = RamSQL_UserUnique[key].userNcall2_str()
        if len(userNcall2_text) > 3000:
            userNcall2_text = userNcall2_text[:3000]
        
        Ruser_counter_text = RamSQL_UserUnique[key].Ruser_counter_str()
        if len(Ruser_counter_text) > 3000:
            Ruser_counter_text = Ruser_counter_text[:3000]

        Muser_counter_text = RamSQL_UserUnique[key].Muser_counter_str()
        if len(Muser_counter_text) > 3000:
            Muser_counter_text = Muser_counter_text[:3000]

        Tag_counter_text = RamSQL_UserUnique[key].tag_counter_str()
        if len(Tag_counter_text) > 3000:
            Tag_counter_text = Tag_counter_text[:3000]

        # get score_fin and Ncall_fin
        if len(RamSQL_UserUnique[key].userScore1) > 0:
            score1_fin = RamSQL_UserUnique[key].userScore1[-1]
        else:
            score1_fin = 0

        if len(RamSQL_UserUnique[key].userNcall1) > 0:  
            Ncall1_fin = RamSQL_UserUnique[key].userNcall1[-1]
        else:
            Ncall1_fin = 0

        if len(RamSQL_UserUnique[key].userScore2) > 0:
            score2_fin = RamSQL_UserUnique[key].userScore2[-1]
        else:
            score2_fin = 0

        if len(RamSQL_UserUnique[key].userNcall2) > 0:  
            Ncall2_fin = RamSQL_UserUnique[key].userNcall2[-1]
        else:
            Ncall2_fin = 0

        # command for Tweet_Stack
        comd_UserUnique_Insert = """
INSERT INTO %s (userID, userName, totalAction, followers, friends, score1_fin, Ncall1_fin, \
score2_fin, Ncall2_fin, userScore1_text, userNcall1_text, userScore2_text, userNcall2_text, \
Ruser_counter_text, Muser_counter_text, Tag_counter_text)
VALUES ( %s, '%s', %s, %s, %s, %s, %s, %s, %s, '%s', '%s', '%s', '%s', '%s', '%s', '%s')
ON DUPLICATE KEY UPDATE totalAction = %s;"""
        # execute commands
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_UserUnique_Insert % tuple( [tableName] + 
                                                                [RamSQL_UserUnique[key].user] + 
                                                                [RamSQL_UserUnique[key].userName] + 
                                                                [str(RamSQL_UserUnique[key].totalAction)] + 
                                                                [str(RamSQL_UserUnique[key].followers_count)] + 
                                                                [str(RamSQL_UserUnique[key].friends_count)] + 
                                                                [str(score1_fin)] + 
                                                                [str(Ncall1_fin)] + 
                                                                [str(score2_fin)] + 
                                                                [str(Ncall2_fin)] + 
                                                                [userScore1_text] + 
                                                                [userNcall1_text] + 
                                                                [userScore2_text] + 
                                                                [userNcall2_text] + 
                                                                [Ruser_counter_text] + 
                                                                [Muser_counter_text] + 
                                                                [Tag_counter_text] + 
                                                                [str(RamSQL_UserUnique[key].totalAction)]
                                                               ) 
                              )
            # commit commands
            connection.commit()
        finally:
            pass


####################################################################

def UserUnique_Convert(connection, pin_time):

    # table Name
    tableName = "UserUnique_"+pin_time.strftime('%Y_%m_%d_%H')
    #Comd
    comd_convert = """
ALTER TABLE %s ENGINE=InnoDB;"""
    # execute commands
    try:
        with connection.cursor() as cursor:
            cursor.execute( comd_convert % tableName )
        # commit commands
        print tableName+" Converted"
        connection.commit()
    finally:
        pass

"""
####################################################################
"""

def RollingScoreBank_Save(connection, pin_time, RollingScoreBank):

    # table Name * 6
    table_tag_key1 = "RSB_tag_key1_"+pin_time.strftime('%Y_%m_%d_%H')
    table_tag_key2 = "RSB_tag_key2_"+pin_time.strftime('%Y_%m_%d_%H')

    table_tag_relev1 = "RSB_tag_relev1_"+pin_time.strftime('%Y_%m_%d_%H')
    table_tag_relev2 = "RSB_tag_relev2_"+pin_time.strftime('%Y_%m_%d_%H')

    table_user_key1 = "RSB_User_key1_"+pin_time.strftime('%Y_%m_%d_%H')
    table_user_key2 = "RSB_User_key2_"+pin_time.strftime('%Y_%m_%d_%H')

    # Comds for Initializing 6 tables in 3 categories
    Comd_RSB_tagKey_Init = """
CREATE TABLE IF NOT EXISTS %s
(
    id MEDIUMINT NOT NULL AUTO_INCREMENT,
    PRIMARY KEY (id),
    tag varchar(255),
    tag_Ncall int
)ENGINE=MEMORY;"""

    Comd_RSB_tagRelev_Init = """
CREATE TABLE IF NOT EXISTS %s
(
    id MEDIUMINT NOT NULL AUTO_INCREMENT,
    PRIMARY KEY (id),
    tag varchar(255),
    tag_Score float,
    tag_Ncall int
)ENGINE=MEMORY;"""

    Comd_RSB_User_Init = """
CREATE TABLE IF NOT EXISTS %s
(
    id MEDIUMINT NOT NULL AUTO_INCREMENT,
    PRIMARY KEY (id),
    userID bigint,
    user_Score float,
    user_Ncall int
)ENGINE=MEMORY;"""

    # Comds for Inserting values into 6 tables
    comd_RSB_tagKey_Insert = """
INSERT INTO %s (tag, tag_Ncall)
VALUES ( '%s', %s);"""

    comd_RSB_tagRelev_Insert = """
INSERT INTO %s (tag, tag_Score, tag_Ncall)
VALUES ( '%s', %s, %s);"""

    comd_RSB_User_Insert = """
INSERT INTO %s (userID, user_Score, user_Ncall)
VALUES ( %s, %s, %s);"""

    # Comds for Converting in-RAM tables into InnoDB tables
    comd_convert = """
ALTER TABLE %s ENGINE=InnoDB;"""

    ###################################################################
    # load table_tag_key1 & 2
    counter = 0
    for table_Name in [ table_tag_key1, table_tag_key2]:
        print "loading RollingScoreBank der %s" % table_Name
        counter += 1
        if counter == 1:
            RSB_key = 'tag_keyword1'
        elif counter == 2:
            RSB_key = 'tag_keyword2'
        # Initialize table
        try:
            with connection.cursor() as cursor:
                cursor.execute( Comd_RSB_tagKey_Init % table_Name )
            # commit commands
            connection.commit()
        finally:
            pass
        # Insert values
        for key in RollingScoreBank[RSB_key]:
            try:
                with connection.cursor() as cursor:
                    cursor.execute( comd_RSB_tagKey_Insert % tuple( [table_Name] + 
                                                                    [str(key)] + 
                                                                    [str(RollingScoreBank[RSB_key][key])]
                                                                  ) 
                                  )
                # commit commands
                connection.commit()
            finally:
                pass
        # Convert tables
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_convert % table_Name )
            # commit commands
            connection.commit()
        finally:
            pass

    ###################################################################
    # load table_tag_relev1 & 2
    counter = 0
    for table_Name in [table_tag_relev1, table_tag_relev2]:
        print "loading RollingScoreBank der %s" % table_Name
        counter += 1
        if counter == 1:
            RSB_key = 'tag_relevant1'
            RSB_keyN = 'tag_relevant1_N'
        elif counter == 2:
            RSB_key = 'tag_relevant2'
            RSB_keyN = 'tag_relevant2_N'
        # Initialize table
        try:
            with connection.cursor() as cursor:
                cursor.execute( Comd_RSB_tagRelev_Init % table_Name )
            # commit commands
            connection.commit()
        finally:
            pass
        # Insert values
        for key in RollingScoreBank[RSB_key]:
            if key in RollingScoreBank[RSB_keyN]:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute( comd_RSB_tagRelev_Insert % tuple( [table_Name] + 
                                                                        [str(key)] + 
                                                                        [str(RollingScoreBank[RSB_key][key])] + 
                                                                        [str(RollingScoreBank[RSB_keyN][key])]
                                                                      ) 
                                      )
                    # commit commands
                    connection.commit()
                finally:
                    pass            
        # Convert tables
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_convert % table_Name )
            # commit commands
            connection.commit()
        finally:
            pass

    ###################################################################
    # load table_user_key1 & 2
    counter = 0
    for table_Name in [table_user_key1, table_user_key2]:
        print "loading RollingScoreBank der %s" % table_Name
        counter += 1
        if counter == 1:
            RSB_key = 'user1'
            RSB_keyN = 'user1_N'
        elif counter == 2:
            RSB_key = 'user2'
            RSB_keyN = 'user2_N'
        # Initialize table
        try:
            with connection.cursor() as cursor:
                cursor.execute( Comd_RSB_User_Init % table_Name )
            # commit commands
            connection.commit()
        finally:
            pass
        # Insert values
        for key in RollingScoreBank[RSB_key]:
            if key in RollingScoreBank[RSB_keyN]:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute( comd_RSB_User_Insert % tuple( [table_Name] + 
                                                                        [str(key)] + 
                                                                        [str(RollingScoreBank[RSB_key][key])] + 
                                                                        [str(RollingScoreBank[RSB_keyN][key])]
                                                                      ) 
                                      )
                    # commit commands
                    connection.commit()
                finally:
                    pass            
        # Convert tables
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_convert % table_Name )
            # commit commands
            connection.commit()
        finally:
            pass

    ####################################################################
    # End of RSB saving operation
    return None

####################################################################

def RollingScoreBank_Load(connection, RollingScoreBank, str_pin_time):
    '''
    str_pin_time: string format of time, appendix to RSB table names
    '''
    # table Name * 6
    table_tag_key1 = "RSB_tag_key1_"+str_pin_time
    table_tag_key2 = "RSB_tag_key2_"+str_pin_time

    table_tag_relev1 = "RSB_tag_relev1_"+str_pin_time
    table_tag_relev2 = "RSB_tag_relev2_"+str_pin_time

    table_user_key1 = "RSB_User_key1_"+str_pin_time
    table_user_key2 = "RSB_User_key2_"+str_pin_time

    ####################################################################
    # Comds for selecting data
    # table_tag_key1 &2
    comd_tag_key = """
select tag, tag_Ncall
from %s"""
    # table_tag_relev1 &2
    comd_tag_relev = """
select tag, tag_Score, tag_Ncall
from %s"""
    # table_user_key1 &2
    comd_user_key = """
select userID, user_Score, user_Ncall
from %s"""

    ####################################################################
    # execute command
    # load table_tag_key1 & 2
    counter = 0
    for table_Name in [ table_tag_key1, table_tag_key2]:
        print "creating RollingScoreBank der %s" % table_Name
        counter += 1
        if counter == 1:
            RSB_key = 'tag_keyword1'
        elif counter == 2:
            RSB_key = 'tag_keyword2'
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_tag_key%table_Name )
                result = cursor.fetchall()
                # loop through all rows of this table
                for entry in result:
                    # print entry  
                    # {u'tag': u'hillary2016forsanity', u'tag_Ncall': 1}
                    # in MySQL, tag format is utf8, but in raw data as ASCII
                    tag_utf = str(entry['tag'])
                    tag_temp=tag_utf.decode("utf-8")
                    tag_ascii=tag_temp.encode("ascii","ignore")
                    RollingScoreBank[RSB_key][tag_ascii] = int(entry['tag_Ncall'])
        finally:
            pass
    # load table_tag_relev1 & 2
    counter = 0
    for table_Name in [table_tag_relev1, table_tag_relev2]:
        print "creating RollingScoreBank der %s" % table_Name
        counter += 1
        if counter == 1:
            RSB_key = 'tag_relevant1'
            RSB_keyN = 'tag_relevant1_N'
        elif counter == 2:
            RSB_key = 'tag_relevant2'
            RSB_keyN = 'tag_relevant2_N'
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_tag_relev%table_Name )
                result = cursor.fetchall()
                # loop through all rows of this table
                for entry in result:
                    tag_utf = str(entry['tag'])
                    tag_temp=tag_utf.decode("utf-8")
                    tag_ascii=tag_temp.encode("ascii","ignore") 
                    RollingScoreBank[RSB_key][tag_ascii] = float(entry['tag_Score'])
                    RollingScoreBank[RSB_keyN][tag_ascii] = int(entry['tag_Ncall'])
        finally:
            pass
    # load table_user_key1 & 2
    counter = 0
    for table_Name in [table_user_key1, table_user_key2]:
        print "creating RollingScoreBank der %s" % table_Name
        counter += 1
        if counter == 1:
            RSB_key = 'user1'
            RSB_keyN = 'user1_N'
        elif counter == 2:
            RSB_key = 'user2'
            RSB_keyN = 'user2_N'
        try:
            with connection.cursor() as cursor:
                cursor.execute( comd_user_key%table_Name )
                result = cursor.fetchall()
                # loop through all rows of this table
                for entry in result:
                    # print entry
                    # {u'user_Score': 6.71354, u'user_Ncall': 12, u'userID': 3024985276L}
                    # where in MySQL, userID is in BIGINT format
                    # but in raw data as string
                    # print str(entry['userID'])
                    RollingScoreBank[RSB_key][str(entry['userID'])] = float(entry['user_Score'])
                    RollingScoreBank[RSB_keyN][str(entry['userID'])] = int(entry['user_Ncall'])
        finally:
            pass

    ####################################################################
    # End of RSB loading operation
    return RollingScoreBank

####################################################################

