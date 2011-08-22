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
    """Initializes the sqlite3 database."""
    import sqlite3 

    if not os.path.exists(settings.DB_FILE):
        conn = sqlite3.connect(settings.DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE item (id integer primary key, item_id text unique, formatted_text text unique, text text unique, sentiment text)''')
    else:
        conn = sqlite3.connect(settings.DB_FILE)
    return conn


def sanitize_text(text):
    """
    Formats text to strip unneccesary verbage then returns a bag of words of text.
    """
    
    if not text: return
    
    text = text.lower()

    for s in settings.IGNORE_STRINGS:
        if s in text: return 


    format_pats = (
        #match, replace with
        ("http:\/\/.*/", ''),
        ("@[A-Za-z0-9_]+", ''),
        ("#[A-Za-z0-9_]+", ''),
        ("^\s+", ''),
        ("\s+", ' '),
        ("(\w)\\1{2,}','\\1\\1") #remove occurences of more than two consecutive repeating characters 
    )

    for pat in format_pats:
        re.sub(p[0], p[1], text)

    #convert html entities
    stripped_text = str(BeautifulStoneSoup(text, convertEntities=BeautifulStoneSoup.HTML_ENTITIES))
    sanitized_text = ''.join([c for c in stripped_text if re.match("[a-z\ \n\t]", c)])
    
    if sanitized_text:
        for emoticon in settings.EMOTICONS:
            try:
                sanitized_text = sanitized_text.encode('ascii')
                sanitized_text = sanitized_text.replace(emoticon, '')
            except:
                return
        
        tokens = bag_of_words(sanitized_text)
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
    

def twitter_feed(sentiment, last_id = None, **kwargs):
    db = synt.db_init()
    cursor = db.cursor()
    if sentiment == 'positive':
        query = ':)'
    if sentiment == 'negative':
        query = ':('
    if not last_id:
        cursor.execute('SELECT item_id FROM item WHERE sentiment=? ORDER BY item_id DESC LIMIT 1',[sentiment])
        last_id = cursor.fetchone()[0]
    url = "http://search.twitter.com/search.json?lang=en&q=%s&since_id=%s" % (query,last_id)
    try:
        data = json.loads(urllib2.urlopen(url).read())
    except Exception,e:
        data = None
        pass
    if data:
        items = data['results']
        cursor.execute('SELECT COUNT() FROM item WHERE sentiment = ?',[sentiment])
        total_rows = cursor.fetchone()[0]
        for item in items:
            text = item['text'].encode('utf8')
            formatted_text = sanitize_text(text)
            if formatted_text:
                item_id = item['id']
                try:
                    cursor.execute('INSERT INTO item VALUES (NULL,?,?,?,?)',[item_id,formatted_text,text,sentiment])
                    last_id = item_id
                    print sentiment,total_rows,text
                    total_rows += 1
                except sqlite3.IntegrityError,e:
                    pass
    db.commit()
    db.close()
    return last_id


