import os, sys, string, sqlite3, cPickle, gzip, re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.util import bigrams
from nltk.featstruct import FeatStruct
from nltk.probability import FreqDist, ConditionalFreqDist
from BeautifulSoup import BeautifulStoneSoup
from nltk import word_tokenize

db_file = 'sample_data.db'

classifier_file = 'classifier.pkl'

use_redis = False

use_gzip = False

num_samples = 200000

max_tokens = 20000

emoticons = [':-L',':L','<3','8)','8-)','8-}','8]','8-]','8-|','8(','8-(','8-[','8-{','-.-','xx','</3',':-{',': )',': (',';]',':{','={',':-}',':}','=}',':)',';)',':/','=/',';/','x(','x)',':D','T_T','O.o','o.o','o_O','o.-','O.-','-.o','-.O','X_X','x_x','XD','DX',':-$',':|','-_-','D:',':-)','^_^','=)','=]','=|','=[','=(',':(',':-(',':,(',':\'(',':-]',':-[',':]',':[','>.>','<.<']

ignore_strings = ['RT',':-P',':-p',';-P',';-p',':P',':p',';P',';p']

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
    formatted_text = re.sub("http:\/\/.*/",'', text)
    formatted_text = re.sub("@[A-Za-z0-9_]+",'', formatted_text)
    formatted_text = re.sub("#[A-Za-z0-9_]+",'', formatted_text)
    formatted_text = re.sub("^\s+",'', formatted_text)
    formatted_text = re.sub("\s+",' ', formatted_text)
    formatted_text = str(BeautifulStoneSoup(formatted_text, convertEntities=BeautifulStoneSoup.HTML_ENTITIES))
    formatted_text = ''.join([c for c in formatted_text.lower() if re.match("[a-z\ \n\t]", c)])
    if formatted_text:
        for emoticon in emoticons:
            try:
                formatted_text = formatted_text.encode('ascii')
                formatted_text = formatted_text.replace(emoticon,'')
            except:
                return False
        tokens = word_tokenize(formatted_text)
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

def get_samples(limit=None,only_type=None):
    db = db_init()
    cursor = db.cursor()
    pos_samples = []
    neg_samples = []
    if not limit:
        limit = get_training_limit()
    elif limit and not only_type:
        limit = (int(limit)/2)
    if not only_type == 'negative':
        cursor.execute("SELECT text,sentiment FROM item where sentiment = 'positive' LIMIT ?",[limit])
        neg_samples = cursor.fetchall()
    if not only_type == 'positive':
        cursor.execute("SELECT text,sentiment FROM item where sentiment = 'negative' LIMIT ?",[limit])
        pos_samples = cursor.fetchall()
    samples = pos_samples + neg_samples
    return samples

def gen_classifier(disk_save=True,num_samples=False,max_tokens=False):
    if not num_samples:
        num_samples = 200000
    if not max_tokens:
        max_tokens = 20000
    samples = get_samples(num_samples)
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    score_fn = BigramAssocMeasures.chi_sq
    top_tokens = {'negative':{},'positive':{}}
    check_tokens = {'negative':{},'positive':{}}
    all_tokens = []
    train_tokens = []
    total_samples = len(samples)
    total_tokens = 0
    processed_tokens = 0
    processed_samples = 0
    for sample in samples:
        processed_samples += 1
        percent = int(processed_samples*100/total_samples)
        sys.stdout.write("\rGenerating Classifier Tokens - Samples: %s/%s - %d%%\r" % (processed_samples,total_samples,percent))
        sys.stdout.flush()
        tokens = sanitize_text(sample[0])
        sentiment = sample[1]
        stemmer = PorterStemmer()
        cleaned_words = set(stemmer.stem(w.lower()) for w in tokens) - set(stopwords.words('english')) - set('')
        for word in cleaned_words:
            word_fd.inc(word.lower())
            label_word_fd[sentiment].inc(word.lower())
        try:
            bigram_finder = BigramCollocationFinder.from_words(cleaned_words)
            bigrams = bigram_finder.nbest(score_fn,100)
        except:
            pass
        if bigrams:
            for bigram in bigrams:
                word_fd.inc(bigram)
                label_word_fd[sentiment].inc(bigram)
        else: 
            bigrams = []
        sample_tokens = bigrams
        for word in cleaned_words:
            sample_tokens.append(word)
        all_tokens.append((dict([(token,True) for token in sample_tokens]),sentiment))
    total_word_count = label_word_fd['negative'].N() + label_word_fd['positive'].N()
    for sentiment in top_tokens.keys():
        for word, freq in label_word_fd[sentiment].iteritems():
            score = score_fn(label_word_fd[sentiment][word],(freq, label_word_fd[sentiment].N()), total_word_count)
            top_tokens[sentiment][word] = score
            limit = max_tokens/2
        top_tokens[sentiment] = sorted(top_tokens[sentiment].iteritems(), key=lambda (w,s): s, reverse=True)[:limit]
    print "\n\r"
    for token_group in all_tokens:
        tokens = token_group[0]
        for token in tokens:
            total_tokens +=1
    check_tokens['negative'] = [token for token,score in top_tokens['negative']]
    check_tokens['positive'] = [token for token,score in top_tokens['positive']]
    for token_group in all_tokens:
            sentiment = token_group[1]
            tokens = token_group[0]
            for token in tokens:
                processed_tokens += 1
                percent = int(processed_tokens*100/total_tokens)
                sys.stdout.write("\rGenerating Optimal Training Set - Tokens: %s/%s - %d%%\r" % (processed_tokens,total_tokens,percent))
                sys.stdout.flush()
                if token in check_tokens[sentiment]:
                    train_tokens.append(({token:True},sentiment))
    print "\n\r"
    classifier = NaiveBayesClassifier.train(train_tokens)
    if disk_save:
        print("Saving classifier to disk as: %s" % classifier_file)
        if use_gzip == True:
            fp = gzip.open(classifier_file,'wb')
        else:
            fp = open(classifier_file,'wb')
        cPickle.dump(classifier,fp)
        fp.close()

def get_classifier(generate=False,use_redis=False):
    classifier = False
    if generate == True:
        classifier = gen_classifier(True,num_samples,max_tokens)
    elif not os.path.exists(classifier_file):
        classifier = gen_classifier(True,num_samples,max_tokens)
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
    bag_of_words = gen_bow(text)
    guess = classifier.classify(bag_of_words)
    return guess

def test(classifier=None,num_samples=None):
    if not num_samples:
        num_samples = 10000
    if not classifier:
        classifier = get_classifier()
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
        nltk_testing_dict.append((dict([(token,True) for token in sample_tokens]),sentiment))
    nltk_accuracy = nltk.classify.util.accuracy(classifier,nltk_testing_dict) * 100
    for sample in samples:
        text = sample[0]
        synt_guess = guess(text,classifier)
        known = sample[1]
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
        total_accuracy = accurate_samples*100.00/num_samples
    classifier.show_most_informative_features(30)
    print("\n\rManual classifier accuracy result: %s%%" % total_accuracy) 
    print('\n\rNLTK classifier accuracy result: %.2f%%' % nltk_accuracy)
