# -*- coding: utf-8 -*-
import unittest
from synt.trainer import train
from synt.guesser import Guesser
from synt import config

class TrainerTestCase(unittest.TestCase):

    def test_train_success(self):
        train('samples.db', 1000, best_features=None, purge=True, redis_db=10)

    def test_train_bestwords_success(self):
        train('samples.db', 1000, best_features=250, purge=True, redis_db=10)

    def test_train_bad_db(self):
        self.assertRaises(ValueError, train, 'xyz123.db', redis_db=10, purge=True)

    def test_train_unsupported_classifier(self):
        self.assertRaises(ValueError, train, 'samples.db',
            classifier_type='xyz', redis_db=10)

class GuesserTestCase(unittest.TestCase):

    def setUp(self):
        train('samples.db', 1000, classifier_type='naivebayes', purge=True,
            redis_db=10)
        self.g = Guesser().guess

    def test_guess_with_text(self):
        score = self.g('some random text')
        self.assertTrue(-1.0 <= score <= 1.0)

    def test_guess_no_text(self):
        score = self.g('')
        self.assertEqual(score, 0.0)

    def test_guess_unicode(self):
        score = self.g("FOE JAPANが粘り強く主張していた避難の権利")
        self.assertTrue(-1.0 <= score <= 1.0)

if __name__ == '__main__':
    unittest.main()
