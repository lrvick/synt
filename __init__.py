from synt.utils.redis_manager import RedisManager
from synt.utils.extractors import best_word_feats, sanitize_text
from synt.utils.collector import twitter_feed

m = RedisManager()
classifier = m.load_classifier()

def guess(text, classifier=classifier):
    """Takes a blob of text and returns the sentiment and confidence score."""

    assert classifier, "Needs a classifier."
    
    bag_of_words = best_word_feats(sanitize_text(text))
    if bag_of_words:
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


