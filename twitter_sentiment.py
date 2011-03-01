import urllib2, json, base64, sys, termios, tty, os 

USER = "your_twitter_user"
PASS = "your_twitter_pass"

negative_file = open("negative.txt","w")
positive_file = open("postive.txt","w")

query_post = str("track="+",".join([q for q in queries]))

def read_key():
    oldattr = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSANOW, oldattr)

httprequest = urllib2.Request('http://stream.twitter.com/1/statuses/filter.json',query_post)
auth = base64.b64encode('%s:%s' % (USER, PASS))
httprequest.add_header('Authorization', "basic %s" % auth)
stream = urllib2.urlopen(httprequest)
for item in stream:
    data = json.loads(item)
    if data.get('user',None):
        os.system("clear")
        tweet_text = data['text'].encode('utf8')
        print('\n')
        print tweet_text
        print('\n')
        print("Please choose sentiment for this tweet:")
        print(" 1 = Positive, 2 = Neutral, 3 = Negative, q = Quit ) > ")
        key = read_key()
        if key == 'q':
            print('\n')
            print("You excaped the dungeon!")
            break
        if key == '1':
            positive_file.write("%s \n" % tweet_text)
            print('\n')
            print("Comment saved as: Positive")
        if key == '2':
            print('\n')
            print("Comment Ignored as: Neutral")
        if key == '3':
            negative_file.write("%s \n" % tweet_text)
            print('\n')
            print("Comment saved as: Negative")
