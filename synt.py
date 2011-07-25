import os
import re
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.util import bigrams
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


def get_tokens(num_samples=None):
    all_tokens = []
    stemmer = PorterStemmer()
    samples = get_samples(num_samples)
    for text,sentiment in samples:
        tokens = sanitize_text(text)
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
    return all_tokens


def get_classifier(num_samples=200000):
    all_tokens = get_tokens(num_samples)
    classifier = NaiveBayesClassifier.train(all_tokens)
    return classifier


def guess(text, classifier=None):
    if not classifier:
        classifier = get_classifier()
    bag_of_words = gen_bow(text)
    guess = classifier.classify(bag_of_words)
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
        total_accuracy = accurate_samples * 100.00 / train_samples
    classifier.show_most_informative_features(30)
    print("\n\rManual classifier accuracy result: %s%%" % total_accuracy)
    print('\n\rNLTK classifier accuracy result: %.2f%%' % nltk_accuracy)


if __name__=="__main__":
    test(500,500)
