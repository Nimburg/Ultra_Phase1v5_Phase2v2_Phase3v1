

import json
import os
import numpy as np 
import pandas as pd
import collections as col
import statistics
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
                MarkedTags_dict[tag_text] = (0.0,0.0)
            # insert value
            # keyword 1
            if counter_keyword == 1:
                MarkedTags_dict[tag_text] = tuple( [ sum(x) for x in zip( MarkedTags_dict[tag_text],
                                                                        ( float(row[1]), 0.0 ) 
                                                                        ) ] 
                                                 )
            # keyword 2
            if counter_keyword == 2:
                MarkedTags_dict[tag_text] = tuple( [ sum(x) for x in zip( MarkedTags_dict[tag_text],
                                                                        ( 0.0 , float(row[1]) ) 
                                                                        ) ] 
                                                 )
    # return dict()
    return MarkedTags_dict


