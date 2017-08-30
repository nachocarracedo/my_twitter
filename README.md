## VISUALIZE A TWITTER ACCOUNT

## SUMMARY

This script will output a visualization of you twitter account. The script will gather the last 200 of all the accounts you follow and it will create a visualization that shows:

* Username, number of followers and friends
* Number of tweets, retweets, and replies
* Wordcloud of tweets, retweets, replies, and all tweets together(This blog post explains how the wordclouds are done: http://peekaboo-vision.blogspot.de/2012/11/a-wordcloud-in-python.html)
* Top 20 hashtags
* Where are the people you follow from? [MAP]
* Sentiment analysis of tweets (neutral tweets count as positive). 


## OUTPUT

As an example, here is the visualization I got for my twitter account:

![Alt text](/images/my_twitter.png?raw=true)


## FILES:

* my_twitter_viz.ipynb: I put all together to get 1 final viz. This is the notebook you should use if you want to get your own tweeter viz.
* twitter.ipynb: Notebook where I used as scratch paper to clean data, EDA, play with the plots settings, and to experiment with different algorithms for sentiment analysis. Also I played a bit with LDA to extract tweet topics.
* viz_your_twitter.py: Script that outputs visualization


## USE 

1) Clone project (https://help.github.com/articles/cloning-a-repository/)

2) Make sure you are running Python 3.X and install libraries in requirements.txt (`pip install -r /path/to/requirements.txt`). Use `pip install libraryname` to install any other library. You'll also need to install stopwords, both for English and Spanish: http://blog.nlpapi.co/how-to-install-nltk-corporastopwords/

3) Get your twitter keys:

* Create a Twitter user account if you do not already have one.
Go to https://apps.twitter.com/ and log in with your Twitter user account. This step gives you a Twitter dev account under the same name as your user account.
* Click “Create New App”
* Fill out the form, agree to the terms, and click “Create your Twitter application”
* In the next page, click on “Keys and Access Tokens” tab, and copy your “API key” and “API secret”. Scroll down and click “Create my access token”, and copy your “Access token” and “Access token secret”. 

4) Add your keys to seetings.py file. Also add extra stopwords so they don't show up in wordclouds.

5) Run script and enjoy your viz!!


## OTHER INFO

* Sentiment analysis uses a logistic regression model and it's only performed on tweets in English. This is the training set: http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/. Other models (Random Forest, SVM) were tested using CV, the best performing model was logistic regression. 
* Next step will be to apply topic modeling (LDA) based on this paper: http://users.cecs.anu.edu.au/~ssanner/Papers/sigir13.pdf
* Script took XXX seconds to run with my twitter account. Getting the language of the tweets takes most of the time.












