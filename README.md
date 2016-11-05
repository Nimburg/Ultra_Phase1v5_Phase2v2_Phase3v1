# Ultra_Phase1v5_Phase2v2

A more in-detail documentation will be created later. 

-----------------------------------------------------------------------------
Objective of this Project:

The objective of this project is to analyze "how individual's preference towards one (or several closely related) subjects might shift under external influence".

The case which was chosen to study is the 2016 presidential election, since this is certainly a major media and public focus point. Twitter was chosen as the data source, primarily due to its free twitter API. 

-----------------------------------------------------------------------------
How this Objective is pursued:

In order to achieved such objective, one has to download and analyze huge amounts of social media data. In this case, I downloaded 18 continuous days' worth of data from twitter API (July 13 to Aug 2, 2016), roughly 120GB. The reason that one needs such a large, continuous data set is because: 1st, since we are trying to understand individual's shifting opinions, we naturally what to have a continuous data set so as not to "miss something"; 2nd, since the fact that individual usually takes some time to have his/her mind changed, and that most people don't spend all day on twitter, thus each individual's activity is rather sparse in time, thus one need to collect data for a reasonably long period of time. 

The Phase1 of this project is devoted to data cleaning and extracting basic relevent information from this huge data set. 

The Phase1v5 is the 5th re-write of Phase1. Since we have the 2016 election as our focus point, thus only a small part of the raw data set are relevent to our purposes. In order to extract these relevent data, I used a dynamic method for data cleaning. The basic idea is that, during the process of scaning through the data set, one will build up a dictionary for hash tags, keywords and individuals that are more closely related to the 2016 elections(tags and individuals have different estimation standards). And we are using this dictionary to extract relevent information. 

The Phase2 of this project is devoted to sentiment analysis. 

During my previous attempts at this subject, I realized that purely statistical variables, though abundent, is not very effective when it comes to figuring out individual's opinion, nor when it comes to classify what kinds of information certain specific individual is receiving. In short, sentiment analysis of tweet text message is absolutely necessary for achieveing the objective. 

The sentiment analysis in Phase2v2 (2nd re-write of Phase2) is performed using the LSTM (Long Short Time Memory) method, written with Theano. Programing implementation wise, this is not particularly challenging. However, what is challenging is how one generates the corpus to train the LSTM network. 

Usually, the "golden standard" corpus when it comes to training a NLP application is a corpus that is manually marked by human beings. Naturally, I don't have time to marked hundreds of thousands of tweets. But the good news is that, since we are studying tweet messages, they all come with one or more hash tags. Thus, I could mark the most frequently used hashtags, and use these hashtags to mark the opinions expressed by tweet messages. Nonetheless, there are still several challenges related to generating a corpus for training. 

1st, of those highly used hash tags, some are neutral, like "trump2016" or "hillary2016". These hash tags are not necessarily used to express support or dislike. Thus one could not use them to mark tweet messages. 

2nd, among hash tags that are clearly biased towards certain sentiments, like "vote4XXXX" or "XXXXislier", those hash tags that express dislike are much more frequently used than those express supported. As a results, the corpus one could get in this way is heavily biased towards the "dislike" sentiment. For both candidates (trump and hillary), the number of tweets expressing dislike is easily ten times larger than that of support. 

![alt tag](https://github.com/Nimburg/Ultra_Phase1v5_Phase2v2/blob/master/tag_july_relevence.gif)

![alt tag](https://github.com/Nimburg/Ultra_Phase1v5_Phase2v2/blob/master/tag_july_HisCall.gif)

-----------------------------------------------------------------------------
The Way Forward

This deposite will be routinely updated for the next one month or so (until I hit a dead end). 

What I intend to do next, is dynamically expand the set of hash tags that I could use for marking tweet messages. 

Also, I will try a "boost with different LSTM as contributing elements". 

-----------------------------------------------------------------------------
Technical Details

This set of codes are written in Python 2.7, using libraries that includes: theano, pymysq and other routine libs like numpy and pandas.

Phase1v5 and Phase2v2 both demands MySQL as the way of managing huge data sets. 

The tokenization scripts are borrowed from https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl. Credits to Josh Schroeder. 

The LSTM training and predictions codes are adapted from the tutorial codes at http://deeplearning.net/tutorial/contents.html. I adapted the training codes so that it would work on my data sets; I added prediction codes so that it would work with other parts of my Phase2v2. 

