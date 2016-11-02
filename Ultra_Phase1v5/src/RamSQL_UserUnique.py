import json
import os
import numpy as np 
import pandas as pd
import collections as col


####################################################################
# intended for two keyword (2 projection axis)
class RamSQL_User:

    def __init__(self, tweetID, Tuser, TuserName, followers_count, friends_count):
        
        # basic items of initialization
        self.user = Tuser
        self.userName = TuserName
        self.totalAction = 1 # totalCall should match length of tweetID_list

        # characteristic user information
        self.followers_count = followers_count
        self.friends_count = friends_count

        # items for keyword 1; 
        # length of tagScore[] should match length of tagNcall
        # length of tagScore[] and tagNcall[] may different from totalCall
        self.userScore1 = []
        self.userNcall1 = []

        # items for keyword 2; length of tagScore[] should match length of tagNcall
        self.userScore2 = []
        self.userNcall2 = []

        # extended item for both keywords
        self.tweetID_list = []
        self.Ruser_counter = col.Counter() # use [key] += 1 directly
        self.Muser_counter = col.Counter()
        self.tag_counter = col.Counter()

        # update tweetID and tweet author
        if tweetID.isdigit() == True:
            self.tweetID_list.append(tweetID)

    ####################################################################

    # should have datatype check in the main program body
    def update_N_score(self, tweetID=None, Ruser=None, Muser=None, tagCon=None):

        # update tweetID_list and totalCall
        # need to further guarantee in main body program that this is NOT updated twice
        if tweetID != None and tweetID.isdigit() == True:
            self.tweetID_list.append(tweetID)
            self.totalAction += 1

        # update reply and mentionuser
        # need to further guarantee in main body program that this user is NOT tweet author
        if Ruser != None and Ruser.isdigit() == True:
            self.Ruser_counter[Ruser] += 1

        if Muser != None and Muser.isdigit() == True:
            self.Muser_counter[Muser] += 1

        # update tag
        if tagCon != None:
            self.tag_counter[tagCon] += 1

    # both scores needed to be updated together
    def update_score(self, score1=0, Ncall1=0, score2=0, Ncall2=0):
        # update score1 and score2
        if score1 != 0 and Ncall1 != 0:
            self.userScore1.append(score1)
            self.userNcall1.append(Ncall1)
        else:
            self.userScore1.append(0)
            self.userNcall1.append(0)

        if score2 != 0 and Ncall2 != 0:
            self.userScore2.append(score2)
            self.userNcall2.append(Ncall2)
        else:
            self.userScore2.append(0)
            self.userNcall2.append(0)

    ####################################################################
    # parse list variables into string

    # parse tweetID_list into string
    def tweetID_list_str(self):
        list_str = ""
        for item in self.tweetID_list:
            list_str = list_str + item + ","
        if len(list_str) > 1:
            list_str = list_str[:-1] # get rid of the last ','
        return list_str

    # parse tagScore1
    def userScore1_str(self):
        list_str = ""
        for item in self.userScore1:
            list_str = list_str + "%.3f,"%round(item,3)
        if len(list_str) > 1:
            list_str = list_str[:-1] # get rid of the last ','
        return list_str

    # parse tagNcall1
    def userNcall1_str(self):
        list_str = ""
        for item in self.userNcall1:
            list_str = list_str + str(item) + ","
        if len(list_str) > 1:
            list_str = list_str[:-1] # get rid of the last ','
        return list_str

    # parse tagScore2
    def userScore2_str(self):
        list_str = ""
        for item in self.userScore2:
            list_str = list_str  + "%.3f,"%round(item,3)
        if len(list_str) > 1:
            list_str = list_str[:-1] # get rid of the last ','
        return list_str

    # parse tagNcall1
    def userNcall2_str(self):
        list_str = ""
        for item in self.userNcall2:
            list_str = list_str + str(item) + ","
        if len(list_str) > 1:
            list_str = list_str[:-1] # get rid of the last ','
        return list_str

    ####################################################################
    # parse dict variables into JSON

    # parse tag_counter
    def tag_counter_str(self):
        dict_str = ""
        for key in self.tag_counter:
            dict_str = dict_str+key+":"+str(self.tag_counter[key])+","
        if len(dict_str) > 1:
            dict_str = dict_str[:-1]
        return dict_str

    # parse user_counter
    def Ruser_counter_str(self):
        dict_str = ""
        for key in self.Ruser_counter:
            dict_str = dict_str+key+":"+str(self.Ruser_counter[key])+","
        if len(dict_str) > 1:
            dict_str = dict_str[:-1]
        return dict_str

    def Muser_counter_str(self):
        dict_str = ""
        for key in self.Muser_counter:
            dict_str = dict_str+key+":"+str(self.Muser_counter[key])+","
        if len(dict_str) > 1:
            dict_str = dict_str[:-1]
        return dict_str

    ####################################################################
    # just in case
    def selfPrint(self):
        print self.user, ", total call: ", self.totalAction, ", len of tweetID_list: ", len(self.tweetID_list)
        print "tweetID_list: ", self.tweetID_list_str()
        
        print "score of keyword1: ", self.userScore1_str()
        print "keyword1 Ncall: ", self.userNcall1_str()
        print "score of keyword2: ", self.userScore2_str()
        print "keyword2 Ncall: ", self.userNcall2_str()
        
        print "reply_user_counter", self.Ruser_counter_str()
        print "mention_user_counter", self.Muser_counter_str()

        print "tag_counter", self.tag_counter_str()


"""
#####################################################################################

test code

#####################################################################################
"""

if __name__ == "__main__":

    RamSQL_UserUnique = col.defaultdict(RamSQL_User)

    ####################################################################
    
    RamSQL_UserUnique['1234'] = RamSQL_User(tweetID='999', Tuser='1234',TuserName='asdf', followers_count=1, friends_count=1)
    RamSQL_UserUnique['1234'].update_score(score1 = 9, Ncall1=1)
    RamSQL_UserUnique['1234'].update_N_score(Muser='5555', tagCon='test')

    RamSQL_UserUnique['1234'].selfPrint()






