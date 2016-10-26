# Ultra_Phase1v5_Phase2v2

A more in-detail documentation will be created later. 

-----------------------------------------------------------------------------
Objective of this Project:

The objective of this project is to analyze "how individual's preference towards one (or several closely related) subjects might shift under external influence".

The case which was chosen to study is the 2016 presidential election, since this is certainly a major media and public focus point. Twitter was chosen as the data source, primarily due to its free twitter API. 

-----------------------------------------------------------------------------
How this Objective is pursued:

In order to achieved such objective, one has to download and analyze huge amounts of social media data. In this case, I downloaded 18 continuous days' worth of data from twitter API (July 13 to Aug 2, 2016), roughly 120GB. The reason that one needs such a large, continuous data set is because: 1st, since we are trying to understand individual's shifting opinions, we naturally what to have a continuous data set so as not to "missing something"; 2nd, since the fact that individual usually takes sometime to have his/her mind changed, and that most people don't spend all day on twitter, thus each individual's activity is rather sparse in time, thus one need to correct data for a reasonably long period of time. 

The Phase1 of this project is devoted to data cleaning and extract basic relevent information from this huge data set. 

The PHase1v5 is the 5th re-write of Phase1. Since we have the 2016 election as our focus point, thus only a small part of the raw data set are relevent to our purposes. In order to extract these relevent data, I used a dynamic method for data cleaning. The basic idea is that, during the process of one scaning through the data set, one will build up a dictionary for hash tags, keywords and individuals that spend more than average of their tweets on topics related to the 2016 elections. And we are using this dictionary to filter out relevent information. 

The Phase2 of this project is devoted to sentiment analysis. 

During my previous attempts on this subject, I realized that purely statistical variables, though abundent, is not very effective when it comes to figuring out individual's opinion, nor when it comes to classify what kinds of information specific individuals are receiving. In short, sentiment analysis of tweet text message is absolutely necessary for achieveing the objective. 

The sentiment analysis in Phase2v2 (2nd re-write of Phase2) is performed using the LSTM (Long Short Time Memory) method, written with Theano. Programing implimentation wise, this is not particularly challenging. However, what is challenging is how one generate the corpus to train the LSTM network. 

Usually, the "golden standard" corpus when it comes to training a NLP application is a corpus that is manually marked by human beings. Naturally, I personally don't have time to marked hundreds of thousands of tweets. But the good news is that, since we are studying tweets messages, they all comes with one or more hash tags. Thus, I could marked the most frequently used hashtags, and use these hashtags to mark the opinions expressed by the tweet messages. Nonetheless, there are still several challenges related to generate a corpus for training. 

1st, of those highly used hash tags, some are neutral, like "trump2016" or "hillary2016". These 




-----------------------------------------------------------------------------
Technical Details

This set of codes are written in Python 2.7, using libraries that includes: theano, pymysql, pandas and other routine libs like numpy

Phase1v5 and Phase2v2 both demands MySQL as the way of managing huge data sets. 

The tokenization scripts are borrowed from 

