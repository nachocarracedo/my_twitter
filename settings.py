APP_KEY = ""
APP_SECRET = ""
OAUTH_TOKEN = ""
OAUTH_TOKEN_SECRET = ""
other_stop_words = ['via','&amp;', 'now','one','thing','us', 
					'will',"it's",'it', 'today','day','take'
					"i'm","u","you","yet", "say","much", "gt",
					"new", "us", "also","don't",'still', "thats",
					"that's"]

try:
	from private import *
except Exception:
	pass