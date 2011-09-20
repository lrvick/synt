import unittest
from synt.trainer import train
from synt.utils import best_word_feats, RedisManager
from synt.guesser import guess

class TrainerTestCase(unittest.TestCase):

    def setUp(self):
        self.man = RedisManager(db='testing')

    def test_train_success(self):
        """
        Test that train completes, classifier stored in redis.
        """
        
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
        
        #set up to get a classifier
        #stores a classifier in redis
        train(
            feat_ex=best_word_feats,
            train_samples=100,
            wordcount_samples=50,
            verbose=False,
        )

    def test_get_classifier(self):
        """
        Test we have a classifier to use.
        """
        
        classifier = self.man.load_classifier()
        self.assertIsNotNone(classifier)

    def test_guess(self):
        """
        Test guess is returning a float between -1 and 1.
        """

        score = guess(u'some random text', classifier=self.man.load_classifier())
        self.assertEqual(type(score), float)

        self.assertTrue(-1 <= score <= 1) 

    def test_guess_nothing(self):
        """
        Test it can handle nothing being passed.
        """
        score = guess('', classifier=self.man.load_classifier())
        self.assertIsNone(score)

    def tearDown(self):
        self.man.r.flushdb()

if __name__ == '__main__':
    unittest.main()
