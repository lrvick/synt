"""Functions that interact with or initial the database."""
import os
import sqlite3

import synt.settings as settings

def db_init(create=True):
    """Initializes the sqlite3 database."""
    if not os.path.exists(os.path.expanduser('~/.synt')):
        os.makedirs(os.path.expanduser('~/.synt/'))

    if not os.path.exists(settings.DB_FILE):
        conn = sqlite3.connect(settings.DB_FILE)
        cursor = conn.cursor()
        if create:
            cursor.execute('''CREATE TABLE item (id integer primary key, item_id text unique, text text unique, sentiment text)''')
    else:
        conn = sqlite3.connect(settings.DB_FILE)
    return conn


def get_sample_limit():
    """
    Returns the limit of samples so that both positive and negative samples
    will remain balanced.

    ex. if returned value is 203 we can be confident in drawing that many samples
    from both tables.
    """

    db = db_init()
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM item where sentiment = 'positive'")
    pos_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM item where sentiment = 'negative'")
    neg_count = cursor.fetchone()[0]
    if neg_count > pos_count:
        limit = pos_count
    else:
        limit = neg_count
    return limit


def get_samples(limit=get_sample_limit(), offset=0):
    """
    Returns a combined list of negative and positive samples.
    """

    db = db_init()
    cursor = db.cursor()

    sql =  "SELECT text, sentiment FROM item WHERE sentiment = ? LIMIT ?"
    sql_with_offset = "SELECT text, sentiment FROM item WHERE sentiment = ? LIMIT ? OFFSET ?"

    if limit < 2: limit = 2

    if limit > get_sample_limit():
        limit = get_sample_limit()

    if not limit % 2 == 0:
        limit -= 1 #we want an even number

    if offset > 0:

        cursor.execute(sql_with_offset, ["negative", limit,offset])
        neg_samples = cursor.fetchall()

        cursor.execute(sql_with_offset, ["positive", limit,offset])
        pos_samples = cursor.fetchall()

    else:

        cursor.execute(sql, ["negative", limit,])
        neg_samples = cursor.fetchall()

        cursor.execute(sql, ["positive", limit,])
        pos_samples = cursor.fetchall()


    return pos_samples + neg_samples
