from synt.utils.collect import twitter_feed
import time
import os
import settings

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

if __name__ == '__main__':
    collect()
