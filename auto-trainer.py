import urllib2, json, base64, sys, os, re, time, sqlite3
from BeautifulSoup import BeautifulStoneSoup

db_file = 'trainer_data.db'

emoticons = ['-.-','xx','<3',':-{',': )',': (',';]',':{','={',':-}',':}','=}',':)',';)',':/','=/',';/','x(','x)',':D','T_T','o.-','O.-','-.o','-.O','X_X','x_x','XD','DX',':-$',':|','-_-','D:',':-)','^_^','=)','=]','=|','=[','=(',':(',':-(',':,(',':\'(',':-]',':-[',':]',':[','>.>','<.<']

ignore_strings = ['RT', ':-P',':-p',';-P',';-p',':P',':p',';P',';p']

if not os.path.exists(db_file):
    conn = sqlite3.connect(db_file)
    curs = conn.cursor()
    curs.execute('''create table item (id integer primary key, item_id text unique, formatted_text text unique, text text unique, sentiment text)''')

def twitter_feed(sentiment, db_fle, last_id = None, **kwargs):
    if sentiment == 'positive':
        query = ':)'
    if sentiment == 'negative':
        query = ':('
    if last_id:
        url = "http://search.twitter.com/search.json?lang=en&q=%s&since_id=%s" % (query,last_id)
    else:
        url = "http://search.twitter.com/search.json?lang=en&q=%s&" % query
    try:
        data = json.loads(urllib2.urlopen(url).read())
    except Exception,e:
        data = None
        pass
    if data:
        items = data['results']
        conn = sqlite3.connect(db_file)
        curs = conn.cursor()
        curs.execute('SELECT COUNT() FROM item WHERE sentiment = ?',[sentiment])
        total_rows = curs.fetchone()[0]
        for item in items:
            text = item['text'].encode('utf8')
            if text:
                for string in ignore_strings:
                    if string in text:
                        text = None
                        break
            if text:
                item_id = item['id']
                formatted_text = re.sub("http:\/\/.*/",'', text)
                formatted_text = re.sub("@[A-Za-z0-9_]+",'', formatted_text)
                formatted_text = re.sub("#[A-Za-z0-9_]+",'', formatted_text)
                formatted_text = re.sub("^\s+",'', formatted_text)
                formatted_text = re.sub("\s+",' ', formatted_text)
                formatted_text = str(BeautifulStoneSoup(formatted_text, convertEntities=BeautifulStoneSoup.HTML_ENTITIES))
                for emoticon in emoticons:
                    formatted_text = formatted_text.replace(emoticon,'')
                try:
                    formatted_text = formatted_text.encode('ascii')
                except:
                    formatted_text = None
                if formatted_text:
                    try:
                        curs.execute('INSERT INTO item VALUES (NULL,?,?,?,?)',[item_id,formatted_text,text,sentiment])
                        last_id = item_id
                        print sentiment,total_rows,formatted_text
                        total_rows += 1
                    except sqlite3.IntegrityError:
                        pass
        conn.commit()
    return last_id

neg_lastid = None
pos_lastid = None

while True:
    time.sleep(2)
    pos_lastid = twitter_feed('positive',db_file,pos_lastid)
    neg_lastid = twitter_feed('negative',db_file,neg_lastid)
