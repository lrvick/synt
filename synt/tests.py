# -*- coding: utf-8 -*-
import unittest
from synt.trainer import train
from synt.guesser import Guesser 
from synt.utils.db import RedisManager

class TrainerTestCase(unittest.TestCase):

    def setUp(self):
        self.man = RedisManager(db=10) #testing db

    def test_train_success(self):
        train('samples.db', 1000, best_features=None, purge=True, redis_db=10)
        self.assertTrue('naivebayes' in self.man.r.keys())
    
    def test_train_bestwords_success(self):
        train('samples.db', 1000, best_features=500, purge=True, redis_db=10)
        self.assertTrue('naivebayes' in self.man.r.keys())
        self.assertTrue('best_words' in self.man.r.keys())

    def test_train_unsupported_classifier(self):
        self.assertEqual(train('samples.db', classifier_type='xyz'), None)

    def test_train_bad_samples(self):
        self.assertEqual(train('samples.db', -2000), None)

    def test_train_bad_db(self):
        self.assertEqual(train('doesntexist', 1000), None)

    def tearDown(self):
        self.man.r.flushdb()

class GuesserTestCase(unittest.TestCase):

    def setUp(self):
        self.man = RedisManager(db=10) #testing db
        train('samples.db', 1000, classifier_type='naivebayes', purge=True)
        self.g = Guesser().guess

    def test_guess_with_text(self):
        score = self.g('some random text')
        self.assertEqual(type(score), float)
        self.assertTrue(-1.0 <= score <= 1.0) 

    def test_guess_no_text(self):
        score = self.g('')
        self.assertEqual(type(score), float)
        self.assertEqual(score, 0.0)

    def test_guess_unicode(self):
        score = self.g("FOE JAPANが粘り強く主張していた避難の権利")
        self.assertEqual(type(score), float)
        self.assertTrue(-1.0 <= score <= 1.0) 

    def tearDown(self):
        self.man.r.flushdb()

if __name__ == '__main__':
    unittest.main()
