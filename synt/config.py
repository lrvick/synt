# -*- coding: utf-8 -*-
"""Config for synt project."""

import os
import nltk

PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))

DB_PATH = "~/.synt/"

EMOTICONS = [
    ':-L', ':L', '<3', '8)', '8-)', '8-}', '8]', '8-]', '8-|', '8(', '8-(',
    '8-[', '8-{', '-.-', 'xx', '</3', ':-{', ': )', ': (', ';]', ':{', '={',
    ':-}', ':}', '=}', ':)', ';)', ':/', '=/', ';/', 'x(', 'x)', ':D', 'T_T',
    'O.o', 'o.o', 'o_O', 'o.-', 'O.-', '-.o', '-.O', 'X_X', 'x_x', 'XD', 'DX',
    ':-$', ':|', '-_-', 'D:', ':-)', '^_^', '=)', '=]', '=|', '=[', '=(', ':(',
    ':-(', ':, (', ':\'(', ':-]', ':-[', ':]', ':[', '>.>', '<.<'
]

CLASSIFIERS = {
    'naivebayes'   : nltk.NaiveBayesClassifier,
    #'maxent'      : nltk.MaximumEntClassifier,
    #'decisiontree': nltk.DecisionTreeClassifier, 
}


