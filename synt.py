import cPickle
import gzip
import nltk
import os
import re
import sqlite3
import string
import sys
import time
from BeautifulSoup import BeautifulStoneSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.featstruct import FeatStruct
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.util import bigrams

db_file = 'synt.db'

classifier_file = 'classifier.pkl'

tokens_file = 'tokens.pkl.gz'

use_redis = False

use_gzip = False

num_samples = 200000

max_tokens = 20000

emoticons = [
    ':-L', ':L', '<3', '8)', '8-)', '8-}', '8]', '8-]', '8-|', '8(', '8-(',
    '8-[', '8-{', '-.-', 'xx', '</3', ':-{', ': )', ': (', ';]', ':{', '={',
    ':-}', ':}', '=}', ':)', ';)', ':/', '=/', ';/', 'x(', 'x)', ':D', 'T_T',
    'O.o', 'o.o', 'o_O', 'o.-', 'O.-', '-.o', '-.O', 'X_X', 'x_x', 'XD', 'DX',
    ':-$', ':|', '-_-', 'D:', ':-)', '^_^', '=)', '=]', '=|', '=[', '=(', ':(',
    ':-(', ':, (', ':\'(', ':-]', ':-[', ':]', ':[', '>.>', '<.<'
]

ignore_strings = ['RT', ':-P', ':-p', ';-P', ';-p', ':P', ':p', ';P', ';p']

try:
    stopwords.words('english')
except Exception, e:
    print e
    nltk.download('stopwords')


def db_init():
    if not os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute('''create table item (id integer primary key, item_id text unique, formatted_text text unique, text text unique, sentiment text)''')
    else:
        conn = sqlite3.connect(db_file)
    return conn


def sanitize_text(text):
    if not text:
        return False
    for string in ignore_strings:
        if string in text:
            return False
    formatted_text = re.sub("http:\/\/.*/", '', text)
    formatted_text = re.sub("@[A-Za-z0-9_]+", '', formatted_text)
    formatted_text = re.sub("#[A-Za-z0-9_]+", '', formatted_text)
    formatted_text = re.sub("^\s+", '', formatted_text)
    formatted_text = re.sub("\s+", ' ', formatted_text)
    formatted_text = str(BeautifulStoneSoup(formatted_text, convertEntities=BeautifulStoneSoup.HTML_ENTITIES))
    formatted_text = ''.join([c for c in formatted_text.lower() if re.match("[a-z\ \n\t]", c)])
    if formatted_text:
        for emoticon in emoticons:
            try:
                formatted_text = formatted_text.encode('ascii')
                formatted_text = formatted_text.replace(emoticon, '')
            except:
                return False
        tokens = gen_bow(formatted_text)
        return tokens
    else:
        return False

def gen_bow(text):
    stemmer = PorterStemmer()
    tokenizer = TreebankWordTokenizer()
    tokens = set(stemmer.stem(x.lower()) for x in tokenizer.tokenize(text)) - set(stopwords.words('english')) - set('')
    bag_of_words = dict([(token, True) for token in tokens])
    return bag_of_words


def get_training_limit():
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


def get_samples(limit=None, only_type=None):
    db = db_init()
    cursor = db.cursor()
    pos_samples = []
    neg_samples = []
    if not limit:
        limit = get_training_limit()
    elif limit and not only_type:
        limit = (int(limit) / 2)
    if not only_type == 'negative':
        cursor.execute("SELECT text, sentiment FROM item where sentiment = 'positive' LIMIT ?", [limit])
        neg_samples = cursor.fetchall()
    if not only_type == 'positive':
        cursor.execute("SELECT text, sentiment FROM item where sentiment = 'negative' LIMIT ?", [limit])
        pos_samples = cursor.fetchall()
    samples = pos_samples + neg_samples
    return samples

def gen_tokens(num_samples=None):
    all_tokens = []
    score_fn = BigramAssocMeasures.chi_sq
    label_word_fd = ConditionalFreqDist()
    samples = get_samples(num_samples)
    total_samples = len(samples)
    processed_samples = 0
    lastout = time.time()
    for text,sentiment in samples:
        processed_samples += 1
        if ((time.time() - lastout) > 0.5):
            percent = int(processed_samples * 100 / total_samples)
            sys.stdout.write("\rGenerating Classifier Tokens - Samples: %s/%s - %d%%\r" % (processed_samples, total_samples, percent))
            sys.stdout.flush()
            lastout = time.time()
        tokens = sanitize_text(text)
        stemmer = PorterStemmer()
        try:
            cleaned_words = set(stemmer.stem(w.lower()) for w in tokens) - set(stopwords.words('english')) - set('')
        except Exception,e:
            print 'Unable to format string %s' % str(sample)
        try:
            bigram_finder = BigramCollocationFinder.from_words(cleaned_words)
            cleaned_words += bigram_finder.nbest(score_fn, 100)
        except:
            pass
        all_tokens.append((dict([(token, True) for token in cleaned_words]), sentiment))
    if use_gzip == True:
        fp = gzip.open(tokens_file,'wb')
    else:
        fp = open(tokens_file,'wb')
    cPickle.dump(all_tokens, fp, protocol=cPickle.HIGHEST_PROTOCOL)
    fp.close()
    return all_tokens

def get_tokens(generate=False,num_samples=None):
    if generate == True:
        print('Generating new tokens from samples')
        tokens = gen_tokens(num_samples)
    elif not os.path.exists(tokens_file):
        print 'Tokens Cache file not found: %s' % tokens_file
        print 'Generating new tokens from samples'
        tokens = gen_tokens(num_samples)
    else:
        total_size = os.path.getsize(tokens_file)
        total_read = 0
        tokens_chunks = str()
        if use_gzip == True:
            fp = gzip.open(tokens_file,'rb')
        else:
            fp = open(tokens_file,'rw')
            tokens = cPickle.loads(fp.read())
    return tokens


def gen_classifier(disk_save=True,num_samples=200000,max_tokens=20000):
    all_tokens = get_tokens(False,max_tokens)
    limit = max_tokens / 2
    score_fn = BigramAssocMeasures.chi_sq
    label_word_fd = ConditionalFreqDist()
    top_tokens = {'negative':{},'positive':{}}
    check_tokens = {'negative':{},'positive':{}}
    train_tokens = []
    total_tokens = 0
    processed_tokens = 0
    for tokens,sentiment in all_tokens:
        for token in tokens:
            label_word_fd[sentiment].inc(token)
    total_word_count = label_word_fd['negative'].N() + label_word_fd['positive'].N()
    for sentiment in top_tokens.keys():
        for word, freq in label_word_fd[sentiment].iteritems():
            score = score_fn(label_word_fd[sentiment][word], (freq, label_word_fd[sentiment].N()), total_word_count)
            top_tokens[sentiment][word] = score
        top_tokens[sentiment] = sorted(top_tokens[sentiment].iteritems(), key=lambda (w, s): s, reverse=True)[:20]
    
    for sentiment in top_tokens.keys():
        for token in top_tokens[sentiment]:
            print sentiment,token
    #for tokens,sentiment in all_tokens:
    #        output_delay = 250
    #        lastout = time.time()
    #        for token in tokens:
    #            processed_tokens += 1
    #            if ((time.time() - lastout) > 0.5):  # Spamming stdout slows the process
    #                percent = int(processed_tokens * 100 / total_tokens)
    #                sys.stdout.write("\rGenerating Optimal Training Set - Tokens: %s/%s - %d%%\r" % (processed_tokens, total_tokens, percent))
    #                sys.stdout.flush()
    #                lastout = time.time()
    #            if token in [token for token,score in top_tokens[sentiment]]:
    #                train_tokens.append(({token:True},sentiment))
    #print "\n\r"
    #classifier = NaiveBayesClassifier.train(train_tokens)
    #if disk_save:
    #    print("Saving classifier to disk as: %s" % classifier_file)
    #    if use_gzip == True:
    #        fp = gzip.open(classifier_file, 'wb')
    #    else:
    #        fp = open(classifier_file, 'wb')
    #    cPickle.dump(classifier, fp)
    #    fp.close()
    #return classifier


def get_classifier(generate=False, use_redis=False):
    classifier = False
    if generate == True:
        classifier = gen_classifier(True, num_samples, max_tokens)
    elif not os.path.exists(classifier_file):
        classifier = gen_classifier(True, num_samples, max_tokens)
    else:
        if use_redis == True:
            if cache.get('synt_class'):
                print ('Loading Classifier from Redis cache')
                classifier = cPickle.loads(cache.get('synt_class'))
        if not classifier:
            print ('Loading Classifier from file: %s' % classifier_file)
            if use_gzip == True:
                fp = gzip.open(classifier_file, 'rb')
            else:
                fp = open(classifier_file, 'rw')
            classifier = cPickle.load(fp)
            fp.close()
            if use_redis == True:
                print ('saving classifier to Redis cache')
                cache.set('synt_class', cPickle.dumps(classifier))
    return classifier


def guess(text, classifier=None):
    if not classifier:
        classifier = get_classifier()
    bag_of_words = gen_bow(text)
    guess = classifier.classify(bag_of_words)
    return guess


def test(classifier=None, num_samples=None):
    if not num_samples:
        num_samples = 10000
    if not classifier:
        print "Classifier not provided, fetching/generating one."
        classifier = get_classifier(20000)
    results_dict = []
    accurate_samples = 0
    samples = get_samples(num_samples)
    #nltk_testing_dict = gen_classifier_dict(samples)
    nltk_testing_dict = []
    for sample in samples:
        sentiment = sample[1]
        stemmer = PorterStemmer()
        tokens = sanitize_text(sample[0])
        sample_tokens = []
        cleaned_words = set(stemmer.stem(w.lower()) for w in tokens) - set(stopwords.words('english')) - set('')
        for word in cleaned_words:
            sample_tokens.append(word)
        nltk_testing_dict.append((dict([(token, True) for token in sample_tokens]), sentiment))
    nltk_accuracy = nltk.classify.util.accuracy(classifier, nltk_testing_dict) * 100
    for sample in samples:
        text = sample[0]
        synt_guess = guess(text, classifier)
        known = sample[1]
        if known == synt_guess:
            accuracy = True
        else:
            accuracy = False
        results_dict.append((accuracy, known, synt_guess, text))
    for result in results_dict:
        print ("Text: %s" % (result[3]))
        print ("Accuracy: %s | Known Sentiment: %s | Guessed Sentiment: %s " % (result[0], result[1], result[2]))
        print ("------------------------------------------------------------------------------------------------------------------------------------------")
        if result[0] == True:
            accurate_samples += 1
        total_accuracy = accurate_samples * 100.00 / num_samples
    classifier.show_most_informative_features(30)
    print("\n\rManual classifier accuracy result: %s%%" % total_accuracy)
    print('\n\rNLTK classifier accuracy result: %.2f%%' % nltk_accuracy)

if __name__=="__main__":
    test()
