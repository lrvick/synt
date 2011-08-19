import re
import redis
import settings

from ntlk.corpus import stopwords
from BeautifulSoup import BeautifulStoneSoup

try:
    stopwords.words('english')
except IOError: #no such file
    nltk.download('stopwords')


def db_init():
    import sqlite3 

    if not os.path.exists(settings.DB_FILE):
        conn = sqlite3.connect(settings.DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''create table item (id integer primary key, item_id text unique, formatted_text text unique, text text unique, sentiment text)''')
    else:
        conn = sqlite3.connect(settings.DB_FILE)
    return conn


def sanitize_text(text):
    """
    Formats text to strip unneccesary verbage then returns a bag of words of text.
    """
    
    if not text: return

    for s in settings.IGNORE_STRINGS:
        if s in text: return 

    
    formatted_text = re.sub("http:\/\/.*/", '', text)
    formatted_text = re.sub("@[A-Za-z0-9_]+", '', formatted_text)
    formatted_text = re.sub("#[A-Za-z0-9_]+", '', formatted_text)
    formatted_text = re.sub("^\s+", '', formatted_text)
    formatted_text = re.sub("\s+", ' ', formatted_text)
    formatted_text = re.sub('(\w)\\1{2,}','\\1\\1', formatted_text) #remove occurence of more than two consecutive repeating chars
    formatted_text = str(BeautifulStoneSoup(formatted_text, convertEntities=BeautifulStoneSoup.HTML_ENTITIES))
    formatted_text = ''.join([c for c in formatted_text.lower() if re.match("[a-z\ \n\t]", c)])
    
    if formatted_text:
        for emoticon in settings.EMOTICONS:
            try:
                formatted_text = formatted_text.encode('ascii')
                formatted_text = formatted_text.replace(emoticon, '')
            except:
                return
        
        tokens = bag_of_words(formatted_text)
        return tokens


def bag_of_words(text):
    """
    Generate bag of words fo that are greater than 1 in length.
    """

    tokens = [w.lower() for w in TreebankWordTokenizer().tokenize(text) if len(w) > 1]
    tokens = set(tokens) - set(stopwords.words('english')) 
    
    return dict([(token, True) for token in tokens]) 


class RedisManager(object):

    def __init__(self):
        self.r = redis.Redis()

    def top_tokens(self, label, start=0, end=10):
        """Return the most popular tokens for label from Redis store."""
         self.r.exists(label):
            return self.r.zrange(label, start, end, withscores=True, desc=True) 
    

