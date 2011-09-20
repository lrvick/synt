"""Tools to deal with text processing."""

import re
import string
import synt.settings as settings
from nltk.tokenize import WhitespaceTokenizer

PUNC_MAP = dict([(ord(x),None) for x in string.punctuation]) 

def sanitize_text(text):
    """
    Formats text to strip unneccesary:words, punctuation and whitespace. Returns a tokenized list.
   
    >>> text = u"ommmmmmg how'r u!? visi t  <html> <a href='http://google.com'> my</a> site @ http://www.coolstuff.com"
    >>> sanitize_text(text)
    [u'ommg', u'howr', u'visi', u'my', u'site', u'httpwwcoolstuffcom']
    """
    
    if not text: return 
    
    text = text.lower()
    
    for e in settings.EMOTICONS:
        text = text.replace(e, '') #remove emoticons
    
    format_pats = (
        #match, replace with
        ("@[A-Za-z0-9_]+", ''), #twitter specific ""
        ("#[A-Za-z0-9_]+", ''), # ""
        ("(\w)\\1{2,}", "\\1\\1"), #remove occurences of more than two consecutive repeating characters
        ("<[^<]+?>", ''), #remove html tags
    )
    
    for pat in format_pats:
        text = re.sub(pat[0], pat[1], text)
   
    if text:
        text = text.translate(PUNC_MAP) #strip punctuation
    
        words = [w for w in WhitespaceTokenizer().tokenize(text) if len(w) > 1]
    
        return words

if __name__ == '__main__':
    import doctest
    doctest.testmod()

