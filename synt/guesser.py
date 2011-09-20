from synt.utils.redis_manager import RedisManager
from synt.utils.extractors import best_word_feats
from synt.utils.text import sanitize_text

MANAGER = RedisManager()
DEFAULT_CLASSIFIER = MANAGER.load_classifier()

def guess(text, classifier=DEFAULT_CLASSIFIER):
    """Takes a blob of text and returns the sentiment and confidence score."""

    assert classifier, "Needs a classifier."
    
    bag_of_words = best_word_feats(sanitize_text(text))
    if bag_of_words:
        guess = classifier.classify(bag_of_words)
        prob = classifier.prob_classify(bag_of_words)
        
        #return a -1 .. 1 score
        score = prob.prob('positive') - prob.prob('negative')
        
        return score
