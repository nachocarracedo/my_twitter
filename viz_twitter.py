import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.basemap import Basemap
plt.style.use('fivethirtyeight')

from wordcloud import WordCloud, STOPWORDS
from twython import Twython
import settings #twitter keys: APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer

import string
from collections import defaultdict

import gensim
from gensim import corpora
import sklearn

import requests
import pickle
import time
import datetime

from langdetect import detect

if __name__ == "__main__":

	start = time.time()
	############### get tweets
	# twython auth
	print("1/8 Collecting tweets ... ")
	twitter = Twython(settings.APP_KEY,
					  settings.APP_SECRET,
					  settings.OAUTH_TOKEN,
					  settings.OAUTH_TOKEN_SECRET)
	# init empty lists to save tweets and metadata
	user_ids, user_names, texts , creation, retweets ,favorites, lenguage, retweet, retweet_from, in_reply, coordinates = ([] for i in range(11))
	# get following IDs and NAMES (can get more info of users if needed!)
	following_ids = [] # to save ids
	following_names = {} # dictinary key:user_id, value: user_name
	user_location = []
	following = twitter.get_friends_ids()["ids"]
	
	############### save details of the account
	#Add details as username and number of people following.
	name = twitter.verify_credentials()['name']
	date_creation = twitter.verify_credentials()['created_at']
	nfollowing = twitter.verify_credentials()['friends_count']
	nfollowers = twitter.verify_credentials()['followers_count']
	
	# get 200 tweets and metadata from each friend (can get more metadata if needed!)
	for user_id in following:
		tweets200 = twitter.get_user_timeline(user_id=user_id,count=200)
		for t in tweets200:
			user_ids.append(user_id)
			texts.append(t["text"])
			creation.append(t["created_at"])
			retweets.append(t["retweet_count"])
			favorites.append(t["favorite_count"])
			lenguage.append(t["lang"])
			in_reply.append(t['in_reply_to_screen_name'])
			retweet.append(('retweeted_status') in t)
			if ('retweeted_status') in t:
				retweet_from.append(t['retweeted_status']['user']['name'])
			else:
				retweet_from.append("N/A")
		#get user location
		user_location.append(twitter.show_user(user_id=user_id)["location"])
		
	# create final DataFrame
	mytweets = pd.DataFrame({'user_id':user_ids,#'user_name':user_names,
							 'text':texts ,
							 'retweet': retweet,
							 'creation':creation,
							 'retweets':retweets ,
							 'favorites':favorites, 
							 'lenguage':lenguage,
							 'retweet_from': retweet_from,
							 'in_reply':in_reply})
	
	# fix in_replay column None type to "None"
	mytweets["in_reply"] = mytweets["in_reply"].map(lambda x: "None" if x is None else x) 
	
	################  Home many of all tweets are regular, tweet, reply
	def tweet_type(row):
		if row.retweet == True:
			return "retweet"
		elif row.in_reply == "None":
			return "reply"
		else:
			return "tweet"
	
	type_tweet = mytweets.apply(tweet_type,1)
	
	
	
	
	print("2/8 Cleaning tweets ... ")
	############### clean tweets
	stop = stopwords.words('english') + stopwords.words('spanish')
	
	stop = set(stop + settings.other_stop_words)
	punctuationset = set(string.punctuation)

	def clean_tweets (tweets_string, punctuation=True, lemmatize=False, stopwords=True):
		""" Gets a string with all tweets and remove RT, links, and ids(start with @).
		OPTIONS: 
		punctuation: if true removes punctuation (true by default)
		lemmatize: if true gets the lemmas of words (false by default)
		stopwords: if true we remove stopwords (true by default). 
		"""
		
		# If it's a a link, a twitter user or RT, we remove it.
		words_clean = " ".join([word.lower() for word in tweets_string.split()
								if 'http' not in word
								and not word.startswith('@')
								and word != 'RT'
								])   
		
		# options
		if stopwords:
			words_clean = " ".join([w for w in words_clean.split() if w not in stop])
			
		#print(words_clean.lower().split())
		if punctuation:
			words_clean = "".join([w for w in words_clean if w not in punctuationset])
			
		if lemmatize : 
				lemma = WordNetLemmatizer() 
				words_clean = " ".join(lemma.lemmatize(word) 
									   for word in words_clean.split()
									   if type(word) is str)
				
		return words_clean

		
	# words of ALL TWEETS
	words = clean_tweets(' '.join(mytweets['text']), lemmatize=False)
	# words of ALL TWEETS keeping # for hashtags
	words_hash = clean_tweets(' '.join(mytweets['text']), lemmatize=False, punctuation=False)
	# words of ALL REGULAR TWEETS 
	words_regular = clean_tweets(' '.join(mytweets[(mytweets["in_reply"]=="None") 
												   & (mytweets["retweet"]==False)].text),
								 lemmatize=False)
	# words of ONLY RETWEETS
	words_rt = clean_tweets(' '.join(mytweets[mytweets["retweet"] == True].text),
							lemmatize=False)
	# words of ONLY REPLIES
	words_reply = clean_tweets(' '.join(mytweets[mytweets["in_reply"] != "None"].text),
							   lemmatize=False)
							   
	print("3/8 Getting languages of tweets ... ")
	############### lenguage of tweets	
	def leng(row):
		text = clean_tweets(row['text'])
		#text = row['text_clean']
		#print(text)
		try:
			return detect(text)
		except Exception as e:
			return 'fail'

	mytweets["lenguage"] = mytweets.apply(leng,1)
	# top 4 languages
	mytweets_len = mytweets[mytweets["lenguage"]!= 'fail']["lenguage"]
	len_top4 = mytweets_len.value_counts()[:4]
		
	print("4/8 Top hashtags ... ")
	############### get top hashtags and words
	def top(xs, top=-1):
		"""Gets words that show up more, option 'top' limits to that number"""
		counts = defaultdict(int)
		for x in xs:
			counts[x] += 1
		return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]

	# hashtags
	top20hash = pd.DataFrame(top([x for x in words_hash.split() if x[0]=='#' and x!='#rt'], top=20),
							 columns=["hashtag", "count"])
	# words
	top20words = pd.DataFrame(top([x for x in words.split()], top=20),
							  columns=["word", "count"])

	
	
	################  location of followings (count)
	print("5/8 Locations ... ")
	locations = top(user_location)
	
	# get more info about the location using google maps API
	list_add=[] # list of lists [localization, number_of_times, latitude, longitude]
	for loc in locations:
		la=[loc[0], loc[1]]
		fixed_address = loc[0].replace(" ","+")
		url = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + fixed_address
		try:        
			response = requests.get(url)
			resp_json_payload = response.json()
			coordinates = resp_json_payload['results'][0]['geometry']['location']
			la.append(str(coordinates['lat'])+" "+str(coordinates['lng']))
			list_add.append(la)
		except:
			continue
			
	location_df = pd.DataFrame(list_add, columns=["location","count", "coordinates"])
	location_plot = location_df.groupby("coordinates")["count"].sum()
	location_plot.sort_values(inplace=True)
		
	
	
	############### Sentiment analysis
	print("6/8 Sentiment analysis ... ")
	# Logistic regression (many models were tested using CV, lg was the best performing one)
	sa_train = pd.read_csv(".\\data\\Sentiment Analysis Dataset.csv",error_bad_lines=False)
	x_sa = sa_train["SentimentText"]
	y_sa = sa_train["Sentiment"]
	del sa_train
	vectorizer = CountVectorizer(min_df=1)
	X = vectorizer.fit_transform(x_sa)
	# This is the model that was fit and saved
	#lge = LogisticRegression(random_state=23, fit_intercept=True, C=0.5, class_weight='balanced')
	# get my data ready for the molel
	mytweets_eng = mytweets[mytweets['lenguage'] == 'en']
	x_unseen = vectorizer.transform(mytweets_eng['text'].map(clean_tweets))
	# load the model from disk
	filename = 'lg_sa_model.sav'
	lgmodel = pickle.load(open('./models/'+filename, 'rb'))
	predictions_unseen = lgmodel.predict(x_unseen)
	prob_unseen = lgmodel.predict_proba(x_unseen)
	results = pd.DataFrame({'Prediction':predictions_unseen,'Prob':list(prob_unseen),'tweet':mytweets_eng['text']})
	
	
	
	print("7/8 Time Series ... ")
	############### number of tweets per date (time series plot)
	mytweets['date'] = pd.to_datetime(mytweets.creation).dt.date
	mytweets['year'] = pd.to_numeric(pd.to_datetime(mytweets.creation).dt.year)
	mytweets['month'] = pd.to_numeric(pd.to_datetime(mytweets.creation).dt.month)
	
	strings = time.strftime("%Y,%m,%d,%H,%M,%S")
	t = strings.split(',')
	numbers = [ int(x) for x in t ]
	
	#only last month worth of tweets
	if numbers[1]==1:
		mytweets_last_month = mytweets[(mytweets['year'] == (numbers[0]-1))& (mytweets['month'] == 12)]
	else:
		mytweets_last_month = mytweets[(mytweets['year'] == numbers[0])& (mytweets['month'] == (numbers[1]-1))] 
		
	tweets_reply = mytweets_last_month[mytweets_last_month.in_reply != None]
	tweets_rt = mytweets_last_month[mytweets_last_month.retweet == True]
	tweets = mytweets_last_month[(mytweets_last_month["in_reply"]=="None")& (mytweets_last_month["retweet"]==False)]

	ts_reply = pd.DataFrame(tweets_reply.groupby(['date']).size()).reset_index()
	ts_reply.columns=['date','count']
	ts_rt = pd.DataFrame(tweets_rt.groupby(['date']).size()).reset_index()
	ts_rt.columns=['date','count']
	ts = pd.DataFrame(tweets.groupby(['date']).size()).reset_index()
	ts.columns=['date','count']
		
		
	print("8/8 Creating viz ... ")	
	################################ VIZ ######################################
	fig = plt.figure(figsize=(15,35))

	
	#type of tweets
	ax1 = plt.subplot2grid((8,2), (0,1))           
	type_tweet.value_counts().plot(kind='bar', color=['#30a2da','#fc4f30','#e5ae38'])
	ax1.set_title("Types of tweets",fontweight = 'bold', size=20)
	plt.xticks(rotation=0)
	plt.margins(0.05)
	 
	 
	#account info
	ax2 = plt.subplot2grid((8,2), (0,0)) 
	
	style1 = dict(size=17, color='black')
	ax2.text(0.15, 0.75, str("Name: "), **style1)
	ax2.text(0.15, 0.55, "Creation: ", **style1)
	ax2.text(0.15, 0.35, "Following: ", **style1)
	ax2.text(0.15, 0.15, "Followers: ", **style1)

	style2 = dict(size=17, color='black',fontweight = 'bold', bbox={'facecolor':'red', 'alpha':0.4, 'pad':8})
	ax2.text(0.45, 0.75, str(name), **style2)
	ax2.text(0.45, 0.55, str(date_creation[4:10].strip()+date_creation[-5:]), **style2)
	ax2.text(0.45, 0.35, str(nfollowing), **style2)
	ax2.text(0.45, 0.15, str(nfollowers), **style2)

	ax2.axis('off')
	ax2.grid(False)
	ax2.set_title("Account information",fontweight = 'bold', size=20)

	
	#type of tweets timeseries
	ax7 = plt.subplot2grid((8,2), (1,0), colspan=2)
	ax7.set_title("Number of tweets by type - 1 month",fontweight = 'bold', size=20)
	ax7.plot_date(ts_reply.date, ts_reply['count'], label='replies',ls='-',marker='.')
	ax7.plot_date(ts_rt.date, ts_rt['count'], label='retweets',ls='-',marker='.')
	ax7.plot_date(ts.date, ts['count'], label='tweets',ls='-',marker='.')
	# set ticks to all days
	plt.gca().xaxis.set_major_locator(mdates.DayLocator())
	# limit axes	
	start_date = ts.date.iloc[0] + datetime.timedelta(days=-1)
	end_date = ts.date.iloc[-1] + datetime.timedelta(days=+1)
	ax7.set_xlim([start_date,end_date])
	ax7.set_ylim(ymin=0)
	# format dates x labels
	xfmt = mdates.DateFormatter('%d-%m')
	ax7.xaxis.set_major_formatter(xfmt)

	# mark weekends
	def find_weekend_indices(datetime_array):
		indices=[]
		for i in range(len(datetime_array)):
			if datetime_array[i].weekday()>=5:
				indices.append(i)
		
		return indices

	def highlight_weekend(weekend_indices,ax):
		i=0
		while i<len(weekend_indices):
			ax7.axvspan(ts.date.iloc[weekend_indices[i]],
						ts.date.iloc[weekend_indices[i]+1],
						facecolor='green',
						edgecolor='none',
						alpha=.15)
			i+=2					
		return None

	# replace month number with letters
	def correct_labels(ax):
		labels = [item.get_text() for item in ax.get_xticklabels()]
		days=[label.split(" ")[0] for label in labels]
		months=["Ja","Fb","Mr","Ap","My","Jn","Jl","Ag","Sp","Oc","Nv","Dc"]
		final_labels=[]
		for i in range(len(days)):
			a=days[i].split("-")
			final_labels.append(a[0]+"\n"+months[int(a[1])-1])
			
		#don't show 1st and last label
		final_labels[0] = ''
		final_labels[-1] = ''
		ax.set_xticklabels(final_labels)
		
	fig.canvas.draw()
	correct_labels(ax7)
	highlight_weekend(find_weekend_indices(ts.date),ax7)

	plt.legend()
	plt.margins(0.05)


	#worldcloud
	title = ["All Tweets", "Regular Tweets","Retweets","Replies"]
	place = [(2,0),(2,1),(3,0),(3,1)]
	i=0 # for plot titles
	for data in [words, words_regular, words_rt, words_reply]:  
		#generate wordcloud
		wordcloud = WordCloud(#stopwords=STOPWORDS,
							  background_color='black',
							  width=1800,
							  height=1400
							 ).generate(data)
		#print wordcloud
		plt.subplot2grid((8,2), place[i])
		plt.imshow(wordcloud,extent=[0,100,0,1], aspect='auto')
		#plt.extend(extent=[0,100,0,1], aspect='auto')
		plt.title(title[i],fontweight = 'bold', size=20)
		plt.axis('off')	   
		
		plt.margins(0)
		plt.tight_layout()
		i+=1

		
	# hashtags barh plot	
	ax3 = plt.subplot2grid((8,2), (4,0), rowspan=2) 	
	bar_heights =  top20hash['count'].values
	bar_positions = np.arange(len(bar_heights))  
	tick_positions = range(0,20) 

	ax3.barh(bar_positions, bar_heights, 0.5, align='center')#, color='red')
	ax3.set_yticks(tick_positions)
	ax3.set_yticklabels(top20hash['hashtag'].values,)
	ax3.set_yticks(tick_positions)
	ax3.set_title("Top 20 Hashtags",fontweight = 'bold', size=20)
	plt.gca().invert_yaxis()
	plt.margins(0.05)
	

	# sentiment analysis
	ax4 = plt.subplot2grid((8,2), (4,1), rowspan=1)
	plt.pie(results.Prediction.value_counts(),
			autopct='%1.1f%%', shadow=False,
			startangle=90, labels=["Positive+Neutral","Negative"])#,colors=["blue","red"])
	plt.title("Sentiment Analysis",fontweight = 'bold', size=20)
	plt.margins(0.05)
	
	
	# languages
	ax5 = plt.subplot2grid((8,2), (5,1), rowspan=1)
	len_top4.plot(kind='bar',color='#6d904f')
	ax5.set_title("Languages",fontweight = 'bold', size=20)
	plt.margins(0.05)
	plt.xticks(rotation=0)

	
	#map
	ax6 = plt.subplot2grid((8,2), (6,0), colspan=2, rowspan=2)
	m = Basemap()
	m.bluemarble() #m.etopo()
	m.drawcountries()

	# cities
	marker_size = [6,10,14,19,24]

	for loc in location_plot.index:
		lat = loc.split()[0]
		lon = loc.split()[1]
		# bin 5 for clusters
		count = int(location_plot.loc[loc]/10)
		if count > 4: count=4
		m.plot(lon,lat,'bo',markersize=marker_size[count], color='red',alpha=0.6,markeredgecolor='black',
			 markeredgewidth=1) 

	labels = ['0-9', '10-19', '20-29', '30-39', '40+']

	leg = plt.legend(labels, ncol=1,fontsize=12, handlelength=2.5, loc="lower left",
					 borderpad = 1.8,handletextpad=1, title='Number of following:', scatterpoints = 1)
	leg.legendHandles[0]._legmarker.set_markersize(6)
	leg.legendHandles[1]._legmarker.set_markersize(9)
	leg.legendHandles[2]._legmarker.set_markersize(12)
	leg.legendHandles[3]._legmarker.set_markersize(15)
	leg.legendHandles[4]._legmarker.set_markersize(18)

	plt.title("Accounts you follow MAP",fontweight = 'bold', size=20)
	plt.margins(0.05)	

	plt.savefig('.\\images\\twitter_viz.png', bbox_inches='tight')		
	print("Done. Find your viz in /images folder")
	print('The script ran for ', time.time()-start, 'seconds.')