import os, sys, sqlite3, cPickle, gzip, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import WordPunctTokenizer
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

db_file = 'sample_data.db'

classifier_file = 'classifier.gz'

use_redis = False

def db_init():
    if not os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        curs = conn.cursor()
        curs.execute('''create table item (id integer primary key, item_id text unique, formatted_text text unique, text text unique, sentiment text)''')
    else:
        conn = sqlite3.connect(db_file)
        curs = conn.cursor()
    return conn.cursor()

def gen_bow(text):
    try:
        stopwords.words('english')
    except Exception, e:
        print e
        nltk.download('stopwords')
    stemmer = PorterStemmer()
    tokenizer = WordPunctTokenizer()
    tokens =  [stemmer.stem(x.lower()) for x in tokenizer.tokenize(text) if x not in stopwords.words('english') and len(x) > 1]
    bag_of_words = dict([(token, True) for token in tokens])
    return bag_of_words

def get_training_limit(): 
    db = db_init()
    db.execute("SELECT COUNT(*) FROM item where sentiment = 'positive'")
    pos_count = db.fetchone()[0]
    db.execute("SELECT COUNT(*) FROM item where sentiment = 'negative'")
    neg_count = db.fetchone()[0]
    if neg_count > pos_count:
        limit = pos_count
    else:
        limit = neg_count
    return limit


def gen_classifier():
    db = db_init()
    limit = get_training_limit()
    db.execute("SELECT * FROM item where sentiment = 'positive' LIMIT ?",[limit])
    samples = db.fetchall()
    db.execute("SELECT * FROM item where sentiment = 'negative' LIMIT ?",[limit])
    samples += db.fetchall()
    sample_num = len(samples)
    train_feats = []
    total_samples = limit*2
    processed_samples = 0
    for sample in samples:
        percent = int(processed_samples*100/total_samples)
        sys.stdout.write("\rGenerating Classifier - Samples: %s/%s - %d%%" % (processed_samples,total_samples,percent))
        sys.stdout.flush()
        processed_samples += 1
        text = sample[2]
        sentiment = sample[4]
        bag_of_words = gen_bow(text)
        if bag_of_words and sentiment:
            train_feats.append((bag_of_words,sentiment))
    classifier = NaiveBayesClassifier.train(train_feats)
    print ('saving classifier')
    fp = gzip.open(classifier_file,'wb')
    cPickle.dump(classifier,fp)
    fp.close()
    return classifier

def get_classifier(generate=False,use_redis=False):
    db = db_init()
    if generate == True:
        classifier = gen_classifier()
    elif not os.path.exists(classifier_file):
        classifier = gen_classifier()
    if use_redis == True:
        if cache.get('synt_class'):
            print ('loading classifier from Redis cache')
            classifier = cPickle.loads(cache.get('synt_class'))
    else:
        print ('loading classifier from file')
        fp = gzip.open(classifier_file,'rb')
        classifier = cPickle.load(fp)
        fp.close()
        print ('saving classifier to Redis cache')
        if use_redis == True:
            cache.set('synt_class',cPickle.dumps(classifier))
    return classifier

def guess(text,classifier=None):
    if not classifier:
        classifier = get_classifier()
    print ('classifier loaded')
    bag_of_words = gen_bow('text')
    return classifier.classify(bag_of_words)

def test():
    db = db_init()
    classifier = get_classifier()
    results_dict = []
    total_accuracy = 0
    db.execute("SELECT * FROM item LIMIT 100")
    samples = db.fetchall()
    for sample in samples:
        text = sample[2]
        print sample
        synt_guess = guess(text,classifier)
        known = sample[4]
        if known == synt_guess:
            accuracy = True
        else:
            accuracy = False
        results_dict.append((accuracy,known,synt_guess,text))
    for result in results_dict:
        print ("Text: %s" % (result[3]))
        print ("Accuracy: %s | Known Sentiment: %s | Guessed Sentiment: %s " % (result[0],result[1],result[2]))
        print ("----------------------------------------------------------------------------------------------")
        if result[0] == True:
            total_accuracy += 1
    print("Total classifier accuracy = %s%%" % total_accuracy) 

test()
