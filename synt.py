import os, sqlite3, cPickle, gzip, redis, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import WordPunctTokenizer
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

db_file = 'sample_data.db'

classifier_file = 'classifier.gz'

def db_init():
    if not os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        curs = conn.cursor()
        curs.execute('''create table item (id integer primary key, item_id text unique, formatted_text text unique, text text unique, sentiment text)''')
    else:
        conn = sqlite3.connect(db_file)
        curs = conn.cursor()
    return conn.cursor()

def gen_classifier():
    try:
        stopwords.words('english')
    except Exception, e:
        print e
        nltk.download('stopwords')
    db = db_init()
    db.execute("SELECT COUNT(*) FROM item where sentiment = 'positive'")
    pos_count = db.fetchone()[0]
    db.execute("SELECT COUNT(*) FROM item where sentiment = 'negative'")
    neg_count = db.fetchone()[0]
    if neg_count > pos_count:
        limit = pos_count
    else:
        limit = neg_count
    db.execute("SELECT * FROM item where sentiment = 'positive' LIMIT ?",[limit])
    samples = db.fetchall()
    db.execute("SELECT * FROM item where sentiment = 'negative' LIMIT ?",[limit])
    samples += db.fetchall()
    sample_num = len(samples)
    train_feats = []
    print ('generating classifier')
    stemmer = PorterStemmer()
    tokenizer = WordPunctTokenizer()
    for sample in samples:
        text = sample[2]
        tokens = tokenizer.tokenize(text)
        word_feats =  [stemmer.stem(x.lower()) for x in tokens if x not in stopwords.words('english') and len(x) > 1]
        sentiment = sample[4]
        if word_feats and sentiment:
            train_feats.append((word_feats,sentiment))
    classifier = NaiveBayesClassifier.train(train_feats)
    print ('saving classifier')
    fp = gzip.open(classifier_file,'wb')
    cPickle.dump(classifier,fp)
    fp.close()
    return classifier

def get_classifier(generate=False):
    db = db_init()
    cache = redis.Redis()
    if generate == True:
        classifier = gen_classifier()
    elif not os.path.exists(classifier_file):
        classifier = gen_classifier()
    elif cache.get('synt_class'):
        print ('loading classifier from Redis cache')
        classifier = cPickle.loads(cache.get('synt_class'))
    else:
        print ('loading classifier from file')
        fp = gzip.open(classifier_file,'rb')
        classifier = cPickle.load(fp)
        fp.close()
        print ('saving classifier to Redis cache')
        cache.set('synt_class',cPickle.dumps(classifier))
    return classifier

def guess(text):
    classifier = get_classifier()
    print ('classifier loaded')
    text_dict = dict([(word,True) for word in text.split(' ')])
    return classifier.classify(text_dict)

def test():
    db = db_init()
    results_dict = []
    db.execute("SELECT * FROM item LIMIT 100")
    samples = db.fetchall()
    for sample in samples:
        text = sample[2]
        print sample
        synt_guess = guess(text)
        known = sample[4]
        if known == synt_guess:
            accuracy = True
        else:
            accuracy = False
        results_dict.append((synt_guess,known,text))
        print (" Known Sentiment: %s | Guessed Sentiment: %s | Text: %s" % (known,synt_guess,text))
   
#print guess('I think cheese is stupid') 

gen_classifier()
