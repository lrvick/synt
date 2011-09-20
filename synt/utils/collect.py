"""Tools for collecting sample data."""

import simplejson as json
import urllib2
import sqlite3
from synt.utils.db import db_init
from synt.utils.text import sanitize_text

def twitter_feed(sentiment, last_id=None, new=False):
    """
    Will return positive and negative tweets from the twitter api.

    What is flagged as positive and negative iscurrently determined
    by happy/sad faces.
    """
    
    db = db_init()
    cursor = db.cursor()

    if sentiment == 'positive': 
        query = ':)'
    elif sentiment == 'negative':
        query = ':('
    else:
        print('Sentiment must be either positive or negative.')
        return

    last_id_url = "http://search.twitter.com/search.json?lang=en&q=%s&since_id=%s"
    query_url   = "http://search.twitter.com/search.json?lang=en&q=%s" 

    if not (last_id or new):
        cursor.execute('SELECT item_id FROM item WHERE sentiment=? ORDER BY item_id DESC LIMIT 1', [sentiment])
        last_id = cursor.fetchone()[0]
        url = last_id_url  % (query, last_id)
    elif last_id:
        url = last_id_url  % (query, last_id)
    elif new:
        url = query_url % query
    
    print(url) 

    data = []

    try:
        data = json.loads(urllib2.urlopen(url).read())
    except:
        raise
    
    if data:
        
        items = data['results']
        cursor.execute('SELECT COUNT() FROM item WHERE sentiment = ?',[sentiment])
        total_rows = cursor.fetchone()[0]
        
        for item in items:
            
            text = unicode(item['text']) #force unicode for db
            
            if text:
                
                item_id = item['id']
                
                try:
                    cursor.execute('INSERT INTO item VALUES (NULL,?,?,?)', [item_id, text, sentiment])
                    last_id = item_id
                    print sentiment, total_rows, text
                    total_rows += 1
                except sqlite3.IntegrityError, e:
                    pass #these are duplicates, we don't want duplicates

    db.commit()
    db.close()

    return last_id
