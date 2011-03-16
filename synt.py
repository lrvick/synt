import os, sys, sqlite3, cPickle, gzip, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import WordPunctTokenizer
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

db_file = 'sample_data.db'

classifier_file = 'classifier.pkl'

use_redis = False

use_gzip = False

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
    tokens = set(stemmer.stem(x.lower()) for x in tokenizer.tokenize(text)) - set(stopwords.words('english')) - set('')
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

def get_samples(limit=None):
    db = db_init()
    if not limit:
        limit = get_training_limit()
    else:
        limit = (int(limit)/2)
    db.execute("SELECT * FROM item where sentiment = 'positive' LIMIT ?",[limit])
    samples = db.fetchall()
    db.execute("SELECT * FROM item where sentiment = 'negative' LIMIT ?",[limit])
    samples += db.fetchall()
    return samples

def gen_classifier_dict(samples=None):
    if not samples:
        samples = get_samples()
    sample_num = len(samples)
    classifier_dict = []
    total_samples = len(samples)
    processed_samples = 0
    for sample in samples:
        percent = int(processed_samples*100/total_samples)
        processed_samples += 1
        sys.stdout.write("\rGenerating Classifier Dict - Samples: %s/%s - %d%%\r" % (processed_samples,total_samples,percent))
        sys.stdout.flush()
        text = sample[2]
        sentiment = sample[4]
        bag_of_words = gen_bow(text)
        if bag_of_words and sentiment:
            classifier_dict.append((bag_of_words,sentiment))
    sys.stdout.write("\rGenerating Classifier Dict - Samples: %s/%s - 100%%\r" % (processed_samples,total_samples))
    sys.stdout.write("\n\r")
    return classifier_dict

def gen_classifier():
    classifier_data = gen_classifier_dict()
    classifier = NaiveBayesClassifier.train(classifier_data)
    print("Saving classifier to disk as: %s" % classifier_file)
    if use_gzip == True:
        fp = gzip.open(classifier_file,'wb')
    else:
        fp = open(classifier_file,'wb')
    cPickle.dump(classifier,fp)
    fp.close()
    return classifier

def get_classifier(generate=False,use_redis=False):
    db = db_init()
    classifier = False
    if generate == True:
        classifier = gen_classifier()
    elif not os.path.exists(classifier_file):
        classifier = gen_classifier()
    else:
        if use_redis == True:
            if cache.get('synt_class'):
                print ('Loading Classifier from Redis cache')
                classifier = cPickle.loads(cache.get('synt_class'))
        if not classifier: 
            print ('Loading Classifier from file: %s' % classifier_file)
            if use_gzip == True:
                fp = gzip.open(classifier_file,'rb')
            else:
                fp = open(classifier_file,'rw')
            classifier = cPickle.load(fp)
            fp.close()
            if use_redis == True:
                print ('saving classifier to Redis cache')
                cache.set('synt_class',cPickle.dumps(classifier))
    return classifier

def guess(text,classifier=None):
    if not classifier:
        classifier = get_classifier()
    bag_of_words = gen_bow('text')
    return classifier.classify(bag_of_words)

def test(num_samples=None):
    if not num_samples:
        num_samples = 1000
    results_dict = []
    accurate_samples = 0
    classifier = get_classifier()
    samples = get_samples(num_samples)
    nltk_testing_dict = gen_classifier_dict(samples)
    nltk_accuracy = nltk.classify.util.accuracy(classifier,nltk_testing_dict) * 100
    for sample in samples:
        text = sample[2]
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
        print ("------------------------------------------------------------------------------------------------------------------------------------------")
        if result[0] == True:
            accurate_samples += 1
        total_accuracy = accurate_samples*100/num_samples
    print("\n\rManual classifier accuracy result: %s%%" % total_accuracy) 
    print('\n\rNLTK classifier accuracy result: %.2f%%' % nltk_accuracy)

test()
