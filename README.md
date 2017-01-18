## VISUALIZE A TWITTER ACCOUNT

## SUMMARY

I wanted to get some information out of my tweeter account, things like:

* Where are the people I follow from? Show in map
* What are the most common words used?
* What about hashtags?
* What kind of tweets are there?
* Sentiment analysis (logistic regression) of tweets (neutral tweets count as positive). Training set from http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/

The goal is to present all this information through visualizations.

The coolest thing about this is that you can run the notebook (my_twitter_viz.ipynb) with your own keys and get the info from your twitter


## FILES:

* my_twitter_viz.ipynb: I put all together to get 1 final viz. This is the notebook you should use if you want to get your own tweeter viz.

* twitter.ipynb: Notebook where I used as scratch paper to clean data, EDA, play with the plots settings, and to experiment with different algorithms for sentiment analysis. Also I played a bit with LDA to extract tweet topics.


## USE 

1) Download my_twitter_viz.ipynb notebook and settings.py

2) Get your twitter keys:

* Create a Twitter user account if you do not already have one.
Go to https://apps.twitter.com/ and log in with your Twitter user account. This step gives you a Twitter dev account under the same name as your user account.
* Click “Create New App”
* Fill out the form, agree to the terms, and click “Create your Twitter application”
* In the next page, click on “Keys and Access Tokens” tab, and copy your “API key” and “API secret”. Scroll down and click “Create my access token”, and copy your “Access token” and “Access token secret”. 

3) Add your keys to seetings.py

4) Run notebook and enjoy your viz!!

## OUTPUT

Here is the visualization I got for my twitter account (200 for each of the accounts I follow):


![Alt text](/images/my_twitter.png?raw=true)


## To be continued ... 

There are several things I still want to do:

* Add details as username and number of people following.
* Sentiment Analysis: divide tweets by language then do SA
* Number of tweets per date (time series plot)
* LDA for topic modeling (code in scratch notebook: twitter.ipynb)









