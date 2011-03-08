import urllib2, json, base64, sys, os, re, time
from BeautifulSoup import BeautifulStoneSoup

positive_filename = "positive.txt"
negative_filename = "negative.txt"

emoticons = [':-{',': )',': (',';]',':{','={',':-}',':}','=}',':)',';)',':/','=/',';/','x(','x)',':D','T_T','o.-','O.-','-.o','-.O','X_X','x_x','XD','DX',':-$',':|','-_-','D:',':-)','^_^','=)','=]','=|','=[','=(',':(',':-(',':,(',':\'(',':-]',':-[',':]',':[','>.>','<.<']

def twitter_feed(query, filename, last_id = None, **kwargs):
    tweet_file = open(filename,"a")
    if last_id:
        url = "http://search.twitter.com/search.json?lang=en&q=%s&since_id=%s" % (query,last_id)
    else:
        url = "http://search.twitter.com/search.json?lang=en&q=%s&" % query
    data = json.loads(urllib2.urlopen(url).read())
    items = data['results']
    for item in items:
        tweet_text = item['text'].encode('utf8')
        tweet_text = re.sub("http:\/\/.*/",'', tweet_text)
        tweet_text = re.sub("@[A-Za-z0-9_]+",'', tweet_text)
        tweet_text = re.sub("#[A-Za-z0-9_]+",'', tweet_text)
        tweet_text = re.sub("^\s+",'', tweet_text)
        tweet_text = re.sub("\s+",' ', tweet_text)
        tweet_text = str(BeautifulStoneSoup(tweet_text, convertEntities=BeautifulStoneSoup.HTML_ENTITIES))
        for emoticon in emoticons:
            tweet_text = tweet_text.replace(emoticon,'')
        if 'RT' in tweet_text:
            return
        elif ':p' in tweet_text:
            return
        elif ':P' in tweet_text:
            return
        else:
            tweet_file.write("%s \n" % tweet_text)
            last_id = item['id']
            print tweet_text
    return last_id

negative_lastid = None
positive_lastid = None

while True:
    time.sleep(2)
    positive_lastid = twitter_feed(':)',positive_filename,positive_lastid)
    negative_lastid = twitter_feed(':(',negative_filename,negative_lastid)
