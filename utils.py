import redis

def get_popular(label, start=0, end=10):
    """Return the most popular tokens for label from Redis store."""
    r = redis.Redis()
    if r.exists(label):
        return r.zrange(label, start, end, withscores=True, desc=True) 
    

