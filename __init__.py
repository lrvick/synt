try:
    stopwords.words('english')
except IOError: #no such file
    nltk.download('stopwords')

def guess(text, classifier=None):
    if not classifier:
        classifier = get_classifier()
    bag_of_words = bag_of_words(text)
    guess = classifier.classify(bag_of_words)
    prob = classifier.prob_classify(bag_of_words)

    #return guess,[(prob.prob(sample),sample) for sample in prob.samples()]
    return guess



