from synt.utils.collect import twitter_feed

def collect():
    """Will continue populating sample database with content."""
    
    neg_lastid, pos_lastid = None, None

    while True:
        time.sleep(1)
        pos_lastid = twitter_feed('positive', pos_lastid)
        neg_lastid = twitter_feed('negative', neg_lastid)


