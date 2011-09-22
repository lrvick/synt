# -*- coding: utf-8 -*-
import unittest
from synt.trainer import train
from synt.utils import best_word_feats, RedisManager
from synt.guesser import guess

class TrainerTestCase(unittest.TestCase):

    def setUp(self):
        self.man = RedisManager(db='testing')

    def test_train_success(self):
        #stores classifier in redis
        train(
            feat_ex=best_word_feats,
            train_samples=100,
            wordcount_samples=50,
            verbose=False,
        )
        self.assertTrue('classifier' in self.man.r.keys())
    
    def tearDown(self):
        self.man.r.flushdb()

class GuesserTestCase(unittest.TestCase):

    def setUp(self):
        self.man = RedisManager(db='testing')
        
        #stores a classifier in redis
        train(
            feat_ex=best_word_feats,
            train_samples=100,
            wordcount_samples=50,
            verbose=False,
        )

    def test_load_classifier(self):
        classifier = self.man.load_classifier()
        self.assertIsNotNone(classifier)

    def test_guess_with_text(self):
        score = guess('some random text', classifier=self.man.load_classifier())
        self.assertEqual(type(score), float)
        self.assertTrue(-1.0 <= score <= 1.0) 

    def test_guess_no_text(self):
        score = guess('', classifier=self.man.load_classifier())
        self.assertEqual(type(score), float)
        self.asserEqual(score, 0.0)

    def test_guess_unicode(self):
        score = guess("FOE JAPANが粘り強く主張していた避難の権利", classifier=self.man.load_classifier())
        self.assertEqual(type(score), float)
        self.assertTrue(-1.0 <= score <= 1.0) 

    def tearDown(self):
        self.man.r.flushdb()

if __name__ == '__main__':
    unittest.main()
