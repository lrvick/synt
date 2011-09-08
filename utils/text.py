"""Tools to deal with text processing."""

import re
import string
import synt.settings as settings
from BeautifulSoup import BeautifulSoup
from nltk.tokenize import WhitespaceTokenizer

def sanitize_text(text):
    """
    Formats text to strip unneccesary:words, punctuation and whitespace. Returns a tokenized list.
    
    >>> text = "ommmmmmmg how'r u!? visi t   my site @ http://www.coolstuff.com"
    >>> sanitize_text(text)
    ['ommg', 'howr', 'visi', 'my', 'site']
    """
    
    if not text: return 
   
    text = text.lower()

    for e in settings.EMOTICONS:
        text = text.replace(e, '') #remove emoticons

    format_pats = (
        #match, replace with
        ("http.*", ''), #strip links
        ("@[A-Za-z0-9_]+", ''), #twitter specific ""
        ("#[A-Za-z0-9_]+", ''), # ""
        ("(\w)\\1{2,}", "\\1\\1"), #remove occurences of more than two consecutive repeating characters
    )
    
    for pat in format_pats:
        text = re.sub(pat[0], pat[1], text)
    
    try:
        text = str(''.join(BeautifulSoup(text).findAll(text=True))) #strip html and force str
    except Exception, e:
        print 'Exception occured:', e
        return


    text = text.translate(string.maketrans('', ''), string.punctuation).strip() #strip punctuation

    if text:
        words = [w for w in WhitespaceTokenizer().tokenize(text) if len(w) > 1]
    
        return words

if __name__ == '__main__':
    import doctest
    doctest.testmod()

