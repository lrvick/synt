from utils.redis import RedisManager
from utils.extractors import best_word_feats, sanitize_text
from utils.collector import twitter_feed

m = RedisManager()
classifier = m.load_classifer()

def guess(text, classifier=classifier):
    """Takes a blob of text and returns the sentiment and confidence score."""

    bag_of_words = best_word_feats(sanitize_text(text))

    guess = classifier.classify(bag_of_words)
    prob = classifier.prob_classify(bag_of_words)
    return guess,[(prob.prob(sample),sample) for sample in prob.samples()]

def collect_samples():
    """Will continue populating sample database with content."""
    
    neg_lastid, pos_lastid = None, None

    while True:
        time.sleep(1)
        pos_lastid = twitter_feed('positive', pos_lastid)
        neg_lastid = twitter_feed('negative', neg_lastid)


