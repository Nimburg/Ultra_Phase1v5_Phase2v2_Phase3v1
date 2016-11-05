# Ultra_Phase1v5_Phase2v2


-----------------------------------------------------------------------------
Objective of this Project:

The objective of this project is to analyze "how individual's preference towards one (or several closely related) subjects might shift under external influence". The value of such objective is almost self-explanatory, as it is not only important in academic settings (for disciplines like sociology, political science, public policy, etc) but also important in industry settings (advertising, financial market strategy, etc).

The case which was chosen to study is the 2016 presidential election, since this is certainly a major media and public focus point. Twitter was chosen as the data source, primarily due to its free twitter API. 


-----------------------------------------------------------------------------
How this Objective is Pursued:

This project is currently divided into three phases, each for distinct purposes. Please check session "Introduction to the DataSet of this Project" below for information regarding to data set used for this project.

The Phase1 of this project is devoted to data cleaning and extracting basic relevent information from data set. 

The Phase1v5 is the 5th re-write of Phase1. Since we have the 2016 election as our focus point, without caring too much about irrelevent topics, thus only a small part of the raw data set are relevent to our purposes. In order to extract these relevent data, I adopted a dynamic method for data cleaning. The basic idea is that, during the process of scaning through the data set, one will build up a dictionary for hash tags, keywords and individuals that are more closely related to the 2016 elections(tags and individuals have different estimation standards). And we are using this dictionary to extract relevent information. 

The Phase2 of this project is devoted to sentiment analysis. 

During my previous attempts at this subject, I realized that purely statistical variables, though abundent, is not very effective when it comes to figuring out individual's opinion, nor when it comes to classify what kinds of information certain specific individual is receiving. In short, sentiment analysis of tweet text message is absolutely necessary for achieveing the objective. 

The sentiment analysis in Phase2v2 (2nd re-write of Phase2) is performed using the LSTM (Long Short Time Memory) method, written with Theano. Programing implementation wise, this is not particularly challenging. However, what is challenging is how one generates the corpus to train the LSTM network. 

Usually, the "golden standard" corpus when it comes to training a NLP application is a corpus that is manually marked by human beings. Naturally, I don't have time to marked hundreds of thousands of tweets. But the good news is that, since we are studying tweet messages, they all come with one or more hash tags. Thus, I could mark the most frequently used hashtags, and use these hashtags to mark the opinions expressed by tweet messages. Nonetheless, there are still several challenges related to generating a corpus for training. 

1st, of those highly used hash tags, some are neutral, like "trump2016" or "hillary2016". These hash tags are not necessarily used to express support or dislike. Thus one could not use them to mark tweet messages. 

2nd, among hash tags that are clearly biased towards certain sentiments, like "vote4XXXX" or "XXXXislier", those hash tags that express dislike are much more frequently used than those express supported. As a results, the corpus one could get in this way is heavily biased towards the "dislike" sentiment. For both candidates (trump and hillary), the number of tweets expressing dislike is easily ten times larger than that of support. 

Phase3 of this project will be built on Phase1 and Phase2 to generate some very interesting results. 

The two pictures below are generated using a 18-days' continous data set took in July 2016. It demonstrates, on a day-to-day basis, the top 10 most used tags' relevence to the two candidates (Trump and Hillary) as well as how the "most used tags in a specific day" fares when one stretches the time window into three weeks. 

The number before each hash tag are the number of its usage during the day which is specified by the date displayed at top middle title. 

![alt tag](https://github.com/Nimburg/Ultra_Phase1v5_Phase2v2/blob/master/tag_july_relevence.gif)

Relevence of each hash tag to a candidate is scaled as: if the hash tag contains keyword "trump"( or "hillary"), then its relevence is set to 10 correspondingly; if the hash tag doesn't contain any keyword but is ALWAYS used with hash tag of relevence 10, then its relevence would stay at 9; the less frequently a hash tag is associated with keyword or hash tags of higher relevence, the lower its relevence score.

![alt tag](https://github.com/Nimburg/Ultra_Phase1v5_Phase2v2/blob/master/tag_july_HisCall.gif)

Here, the "cumulated number of call" is the cumulated number of a hash tag's usage during the 18-day time window. Naturally, this cumulated number will increase as one moves to a later date. 


-----------------------------------------------------------------------------
Introduction to the DataSet of this Project:

In order to achieved such objective, one has to download and analyze huge amounts of social media data. The reason that one needs such a large, continuous data set is because: 1st, since we are trying to understand individual's shifting opinions, we naturally what to have a continuous data set so as not to "miss something"; 2nd, since the fact that individual usually takes some time to have his/her mind changed, and that most people don't spend all day on twitter, thus each individual's activity is rather sparse in time, thus one need to collect data for a reasonably long period of time.

In this case, I have so far collected (commented on Nov 5th, 2016) 28 days' worth of data, among which 18 contineous days in July (July 13th to Aug 2nd, 2016), 7 contineous days in October (Oct 15th to 21st, 2016) and attempting a contineous data set from Oct 31st til some time after Nov8th election day. On average, each day's worth of data is ~7GB with 3 million tweets. I am only collecting tweets with geo-information indicating its user located within the United States. 

It is worth noting that 3 million tweets per day is apparently too few for such a user base. My guess is that the amount of tweets I could collect is limited by my twitter API query speed. However, since tweets are downloaded randomly (those that are got/missed by my API), it won't have any negtive effects on my analysis. 


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

