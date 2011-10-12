# -*- coding: utf-8 -*-
import time
import os
import bz2
import urllib2
import simplejson as json
from sqlite import IntegrityError
from cStringIO import StringIO
from synt.utils.db import db_init
from synt.logger import create_logger
from synt import settings

logger = create_logger(__file__)

#TODO: twitter_feed should be replaced by kral
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
                except IntegrityError, e:
                    pass #these are duplicates, we don't want duplicates

    db.commit()
    db.close()

    return last_id

def collect():
    """
    Will continuously populate the sample database if it exists
    else it will create a new one.
    """

    neg_lastid, pos_lastid = None, None

    if not os.path.exists(settings.DB_FILE):
        pos_lastid = twitter_feed('positive', pos_lastid, new=True)
        neg_lastid = twitter_feed('negative', neg_lastid, new=True)


    while True:
        time.sleep(1)
        try:
            pos_lastid = twitter_feed('positive', pos_lastid)
            neg_lastid = twitter_feed('negative', neg_lastid)
        except:
            raise

def import_progress():
    global logger, output_count, prcount
    try:
        prcount
        output_count
    except:
        prcount=0
        output_count = 500000
    prcount += 20
    output_count += 20
    if output_count >= 500000:
        output_count = 0
        percent = round((float(prcount) / 40423300 )*100, 2)
        logger.info("Processed %s of 40423300 records (%0.2f%%)" % (prcount,percent))
    return 0

def fetch(verbose=True):
    """
    Pre-populates training database from public archive of ~2mil tweets
    """

    if not verbose:
        logger.setLevel(0)

    response = urllib2.urlopen('https://github.com/downloads/Tawlk/synt/sample_data.bz2')

    total_bytes = int(response.info().getheader('Content-Length').strip())
    saved_bytes = 0
    start_time = time.time()
    last_seconds = 0
    last_seconds_start = 0
    data_buffer = StringIO()

    decompressor = bz2.BZ2Decompressor()

    if os.path.exists(settings.DB_FILE):
        os.remove(settings.DB_FILE)

    db = db_init(create=False)
    db.set_progress_handler(import_progress,20)

    while True:
        seconds = (time.time() - start_time)
        chunk = response.read(8192)
        if not chunk:
            break
        saved_bytes += len(chunk)
        data_buffer.write(decompressor.decompress(chunk))
        if seconds > 1:
            percent = round((float(saved_bytes) / total_bytes)*100, 2)
            speed = round((float(total_bytes / seconds ) / 1024),2)
            speed_type = 'Kb/s'
            if speed > 1000:
                speed = round((float(total_bytes / seconds ) / 1048576),2)
                speed_type = 'Mb/s'
            if last_seconds >= 0.5:
                last_seconds = 0
                last_seconds_start = time.time()
                logger.info("Downloaded %d of %d Mb, %s%s (%0.2f%%)\r" % (saved_bytes/1048576, total_bytes/1048576, speed, speed_type, percent))
            else:
                last_seconds = (time.time() - last_seconds_start)
        if saved_bytes == total_bytes:
            logger.info("Downloaded %d of %d Mb, %s%s (100%%)\r" % (saved_bytes/1048576, total_bytes/1048576, speed, speed_type))
            try:
                db.executescript(data_buffer.getvalue())
            except Exception, e:
                logger.error("Sqlite3 import failed with: %s" % e)
                break


if __name__ == '__main__':
    collect()

