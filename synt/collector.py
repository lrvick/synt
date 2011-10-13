# -*- coding: utf-8 -*-
import time
import os
import bz2
import urllib2
from sqlite3 import IntegrityError
from cStringIO import StringIO
from synt.utils.db import db_init
from synt import settings

from kral import stream

def collect(commit_every=200, max_save=100000):
    """
    Will continuously populate the sample database if it exists
    else it will create a new one.
    
    Keyword Arguments:
    commit_every    -- by default will commit every 100 executes
    max_save        -- will stop collecting at this number
    """
    
    db = db_init()
    cursor = db.cursor()

    queries = {
        ':)' : 'positive',
        ':(' : 'negative',
    }

    print("Collection on {} queries.".format(queries.keys()))
    g = stream(query_list=queries.keys(), service_list="twitter") 

    c = 0
    
    for item in g:
        
        text = unicode(item['text'])
     
        sentiment = queries.get(item['query'], None)

        if sentiment:
            try:
                cursor.execute('INSERT INTO item VALUES (NULL,?,?)', [text, sentiment])
                c += 1
                if c % commit_every == 0: 
                    db.commit()
                if c == max_save:
                    break
                
                print("Collected {} samples to database.".format(c))
            except IntegrityError:
                continue 
    
    db.close()

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
        print("Processed %s of 40423300 records (%0.2f%%)" % (prcount,percent))
    return 0

def fetch():
    """
    Pre-populates training database from public archive of ~2mil tweets
    """
    
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
                print("Downloaded %d of %d Mb, %s%s (%0.2f%%)\r" % (saved_bytes/1048576, total_bytes/1048576, speed, speed_type, percent))
            else:
                last_seconds = (time.time() - last_seconds_start)
        if saved_bytes == total_bytes:
            print("Downloaded %d of %d Mb, %s%s (100%%)\r" % (saved_bytes/1048576, total_bytes/1048576, speed, speed_type))
            try:
                db.executescript(data_buffer.getvalue())
            except Exception, e:
                print("Sqlite3 import failed with: %s" % e)
                break


if __name__ == '__main__':
    collect()

