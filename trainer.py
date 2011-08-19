import os
import re
import sqlite3
import nltk
from redis import Redis
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.probability import DictionaryProbDist, ELEProbDist, FreqDist
from nltk.tokenize.treebank import TreebankWordTokenizer
from collections import defaultdict
from BeautifulSoup import BeautifulStoneSoup

db_file = 'sample_data.db'

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
    """
    Formats text to strip unneccesary verbage then returns a bag of words of text.
    """
    
    if not text: return

    for s in ignore_strings:
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
        for emoticon in emoticons:
            try:
                formatted_text = formatted_text.encode('ascii')
                formatted_text = formatted_text.replace(emoticon, '')
            except:
                return
        
        tokens = gen_bow(formatted_text)
        return tokens

def gen_bow(text):
    """
    Generate bag of words for words that are greater than 1 in length.
    """

    tokenizer = TreebankWordTokenizer()
    tokens = set(x.lower() for x in tokenizer.tokenize(text) if len(x) > 1) - set(stopwords.words('english'))
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


def get_samples(limit=get_training_limit(), offset=0):
    """
    Returns a combined list of negative and positive samples.
    """
    
    db = db_init()
    cursor = db.cursor()
    
    limit = limit / 2  
    offset = offset / 2

    cursor.execute("SELECT text, sentiment FROM item WHERE sentiment = 'negative' LIMIT ? OFFSET ?", [limit,offset])
    neg_samples = cursor.fetchall()
    
    cursor.execute("SELECT text, sentiment FROM item WHERE sentiment = 'positive' LIMIT ? OFFSET ?", [limit,offset])
    pos_samples = cursor.fetchall()

    return pos_samples + neg_samples


def get_tokens(num_samples=None,offset=None):
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



def train(num_samples=500, stepby=1000):
    """
    Mimicks the train method of the NaiveBayesClassiffier but
    stores it to a peristent Redis datastore.
    """

    r = Redis()
    r.flushdb()
    print("Flushed Redis DB")

    labels = ['negative','positive'] 
    samples_left = num_samples
    offset = 0
    while samples_left > 0:
        if samples_left > stepby:
            samples_set = stepby
            samples_left -= stepby
            offset = num_samples - samples_left
        else:
            samples_set = samples_left
            samples_left = 0
        print samples_left,num_samples
        labeled_featuresets = get_tokens(samples_set,offset)
        if labeled_featuresets:
            feature_freqdist = defaultdict(FreqDist)
            fnames = set()
            for featureset, label in labeled_featuresets:
                for fname, fval in featureset.items(): 
                    feature_freqdist[label, fname].inc(fval) 
                    fnames.add(fname)
            for label in labels:
                for fname in fnames:
                    count = feature_freqdist[label, fname].N()
                    if count > 0:
                        prev_score = r.zscore(label, fname)
                        r.zadd(label, fname, count if not prev_score else count + prev_score)
                        print "Label: %s | Fname: %s | Count: %s" %(label,fname,count)


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
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = ELEProbDist(freqdist, bins=len(feature_values[fname]))
        feature_probdist[label,fname] = probdist
    classifier = NaiveBayesClassifier(label_probdist,feature_probdist)
    return classifier


def guess(text, classifier=None):
    if not classifier:
        classifier = get_classifier()
    bag_of_words = gen_bow(text)
    guess = classifier.classify(bag_of_words)
    prob = classifier.prob_classify(bag_of_words)

    #return guess,[(prob.prob(sample),sample) for sample in prob.samples()]
    return guess


def test(train_samples=200000,test_samples=200000):
    results_dict = []
    nltk_testing_dict = []
    accurate_samples = 0
    print "Building Classifier with %s Training Samples" % train_samples
    classifier = get_classifier(train_samples)
    print "Preparing %s Testing Samples" % test_samples
    samples = get_samples(test_samples)
    for sample in samples:
        sentiment = sample[1]
        tokens = sanitize_text(sample[0])
        sample_tokens = []
        cleaned_words = set(w.lower() for w in tokens) - set(stopwords.words('english')) - set('')
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
        total_accuracy = (accurate_samples * 100.00 / train_samples) * 100
    classifier.show_most_informative_features(30)
    print("\n\rManual classifier accuracy result: %s%%" % total_accuracy)
    print('\n\rNLTK classifier accuracy result: %.2f%%' % nltk_accuracy)


if __name__=="__main__":
    #test(50000,500)
    train(2000000, stepby=10000)
    #get_classifier(500)

