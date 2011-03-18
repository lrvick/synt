import urllib2, json, base64, sys, os, re, time, synt, sqlite3
from synt import sanitize_text

def twitter_feed(sentiment, last_id = None, **kwargs):
    db = synt.db_init()
    cursor = db.cursor()
    if sentiment == 'positive':
        query = ':)'
    if sentiment == 'negative':
        query = ':('
    if not last_id:
        cursor.execute('SELECT item_id FROM item WHERE sentiment=? ORDER BY item_id DESC LIMIT 1',[sentiment])
        last_id = cursor.fetchone()[0]
    url = "http://search.twitter.com/search.json?lang=en&q=%s&since_id=%s" % (query,last_id)
    try:
        data = json.loads(urllib2.urlopen(url).read())
    except Exception,e:
        data = None
        pass
    if data:
        items = data['results']
        cursor.execute('SELECT COUNT() FROM item WHERE sentiment = ?',[sentiment])
        total_rows = cursor.fetchone()[0]
        for item in items:
            text = item['text'].encode('utf8')
            formatted_text = sanitize_text(text)
            if formatted_text:
                item_id = item['id']
                try:
                    cursor.execute('INSERT INTO item VALUES (NULL,?,?,?,?)',[item_id,formatted_text,text,sentiment])
                    last_id = item_id
                    print sentiment,total_rows,text
                    total_rows += 1
                except sqlite3.IntegrityError,e:
                    pass
    db.commit()
    db.close()
    return last_id

neg_lastid = None
pos_lastid = None

while True:
    time.sleep(1)
    pos_lastid = twitter_feed('positive',pos_lastid)
    neg_lastid = twitter_feed('negative',neg_lastid)
