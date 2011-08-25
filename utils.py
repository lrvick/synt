import re
import redis
import string
import settings
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import WhitespaceTokenizer
from BeautifulSoup import BeautifulSoup, BeautifulStoneSoup
import itertools
import cPickle as pickle

def word_feats(words):
    """Basic word features, simple bag of words model"""

    return dict([(word, True) for word in words])

def stopword_word_feats(words):
    """Word features with filtered stopwords"""

    stopset = set(stopwords.words('english'))
    return dict([(word,True) for word in words if word not in stopset])

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    """Word features with bigrams"""

    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

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
    Formats text to strip unneccesary:words, punctuation and whitespace. Returns a tokenized set.
    """
    
    if not text: return 
   
    text = text.lower()

    for e in settings.EMOTICONS:
        text = text.replace(e, '') #remove emoticons

    try:
        text = str(''.join(BeautifulSoup(text).findAll(text=True))) #strip html and force str
    except Exception, e:
        print 'Exception occured:', e
        return

    format_pats = (
        #match, replace with
        ("http:\/\/.*/", ''), #strip links
        ("@[A-Za-z0-9_]+", ''), #twitter specific ""
        ("#[A-Za-z0-9_]+", ''), # ""
        ("(\w)\\1{2,}", "\\1\\1"), #remove occurences of more than two consecutive repeating characters
    )
    
    for pat in format_pats:
        text = re.sub(pat[0], pat[1], text)

    text = text.translate(string.maketrans('', ''), string.punctuation).strip() #strip punctuation

    if text:
        words = [w for w in WhitespaceTokenizer().tokenize(text) if len(w) > 1]
    
        return bigram_word_feats(words)

def get_sample_limit():
    """
    Makes sure to return an equivalent amount of negative and positive samples.
    """
    
    db = db_init()
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM item where sentiment = 'positive'")
    pos_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM item where sentiment = 'negative'")
    neg_count = cursor.fetchone()[0]
    if neg_count > pos_count:
        limit = pos_count
    else:
        limit = neg_count
    return limit


def get_samples(limit=2, offset=0):
    """
    Returns a combined list of negative and positive samples.
    """
    
    db = db_init()
    cursor = db.cursor()
   
    sql =  "SELECT text, sentiment FROM item WHERE sentiment = ? LIMIT ?"
    sql_with_offset = "SELECT text, sentiment FROM item WHERE sentiment = ? LIMIT ? OFFSET ?"
    
    
    if limit < 2: limit = 2
    limit = limit / 2  
    
    if offset > 0: 

        cursor.execute(sql_with_offset, ["negative", limit,offset])
        neg_samples = cursor.fetchall()
    
        cursor.execute(sql_with_offset, ["positive", limit,offset])
        pos_samples = cursor.fetchall()

    else:
 
        cursor.execute(sql, ["negative", limit,])
        neg_samples = cursor.fetchall()
        
        cursor.execute(sql, ["positive", limit,])
        pos_samples = cursor.fetchall()
       

    return pos_samples + neg_samples


def get_tokens(num_samples, offset=0):
    """
    Returns a list of sanitized tokens and sentiment.
    """

    all_tokens = []
    samples = get_samples(num_samples, offset)
    for text,sentiment in samples:
        tokens = sanitize_text(text)
        if tokens:
            all_tokens.append((dict([(token, True) for token in tokens]), sentiment))
    return all_tokens


def get_classifier(num_samples=200000):
    labeled_featuresets = get_tokens(num_samples)
    label_freqdist = FreqDist()
    feature_freqdist = defaultdict(FreqDist)
    feature_values = defaultdict(set)
    fnames = set()
    for featureset, label in labeled_featuresets: 
              label_freqdist.inc(label) 
              for fname, fval in featureset.items(): 
                  feature_freqdist[label, fname].inc(fval) 
                  feature_values[fname].add(fval) 
                  fnames.add(fname)
    for label in label_freqdist:
        num_samples = label_freqdist[label]
        for fname in fnames:
            count = feature_freqdist[label, fname].N()
            feature_freqdist[label, fname].inc(None, num_samples-count)
            feature_values[fname].add(None)
    label_probdist = ELEProbDist(label_freqdist)
    print label_probdist
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = ELEProbDist(freqdist, bins=len(feature_values[fname]))
        feature_probdist[label,fname] = probdist
    
    print probdist,feature_probdist
    #classifier = NaiveBayesClassifier(label_probdist,feature_probdist)
    #return classifier


def get_classifier_redis(num_samples=200000):
    """
    Builds a classifier from Redis.
    """
    
    label_freqdist = FreqDist()
    feature_freqdist = defaultdict(FreqDist)
    
    top_neg_features = top_tokens('negative', end=num_samples)
    top_pos_features = top_tokens('positive', end=num_samples)

    label_freqdist.inc('positive')
    label_freqdist.inc('negative')

    for pos_items,neg_items in zip(top_pos_features,top_neg_features):
        feature_freqdist[('positive', pos_items[0])].inc(None, count=pos_items[1])
        feature_freqdist[('negative', neg_items[0])].inc(None, count=neg_items[1])


    label_probdist = ELEProbDist(label_freqdist)
    print label_probdist
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = ELEProbDist(freqdist)
        feature_probdist[label,fname] = probdist

    print probdist, feature_probdist


    #classifier = NaiveBayesClassifier(label_probdist, feature_probdist)
    #return classifier


def get_probdist(label_freqdist, feature_freqdist, feature_values):
    
    label_probdist = ELEProbDist(label_freqdist)
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = ELEProbDist(freqdist)
        feature_probdist[label,fname] = probdist


class RedisManager(object):

    def __init__(self):
        self.r = redis.Redis()

    def store_word_freqdist(samples, stepby=1000):
        """
        Stores a word freqdist to Redis expects a list of samples in the form (text, sentiment)
        ex. (u'This is a text string', 'neg')
        """
        
        offset = 0
        samples_left = samples

        #while samples > 0:
        #    if samples > stepby:
        #        samples 

    def store_classifier(self, classifier, name='classifier'):
        """
        Stores a pickled a classifier into Redis.
        """
        dumped = pickle.dumps(classifier, protocol=pickle.HIGHEST_PROTOCOL)
        self.r.set(name, dumped)
        

    def load_classifier(self, name='classifier'):
        """
        Loads (unpickles) a classifier from Redis.
        """
        loaded = pickle.loads(self.r.get(name))
        return loaded


    def top_tokens(self, label, start=0, end=10):
        """Return the most popular tokens for label from Redis store."""
        if self.r.exists(label):
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

