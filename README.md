# Synt #

  <http://github.com/Tawlk/synt>

## About ##

  Synt (pronounced: "cent") is a python library aiming to be a general
  solution to identifying a given peice of text, particularly social
  network statuses, as either negative, neutral, or positive.

## Current Features ##

  * Can create training data from random twitter statuses
  * Uses emoticons to automatically sort out positive/negative training samples
  * Includes a manual trainer to classify posts by hand
  * Ability to granularly generate classifiers by adjusting sample/token/entropy levels

## Requirements ##

  * A running [Redis](http://redis.com) server
  * pip
  * virtualenv (recommended)
  * python 2.7 
    * sqlite3 issue was fixed in 2.7 [issue](http://code.google.com/p/pysqlite/source/detail?r=9e3fa82223b89ca4e7f9eadedc1297ab5c3eebd9)
    * argparse
  * PyYAML==3.09 
    * unfortnatley nltk requires pyyaml before it can be installed [bug](http://code.google.com/p/nltk/issues/detail?id=508)
    

## Usage / Installation ##

  1. Grab the latest synt:

    ```bash
    pip install -e git+https://github.com/Tawlk/synt/#egg=synt
    ```

  2. Grab the sample database to train on (or build one (below)):

    ```bash
    synt collect fetch
    ```

    If you'd prefer to build a fresh sample db and have the time, just run collect with
    the desired number of recent posts you wish to fetch.

    ```bash
    synt collect 1000
    ```

    **Note**: This also can update an existing database, if you decided to
    initially wget the db but want to add more content the above should still
    work.


  3. Build classifier

    ```bash
    synt train
    ```

    Note this requires Redis to be started!


  4. Usage:

    ```python
    from synt import guess
    guess('My mother in law makes me do bad things and I think pandas are icky!')
    ```
    Or on the command line:

    ```bash
    synt guess 'I want to chase poodles. That makes me happy and joyful :-D'
    ```

    This will return a score from -1 .. 1 negative to positive respectivley.

    Anything close to 0 should be considered neutral.


## Notes ##

  Use at your own risk. You may be eaten by a grue.

  Questions/Comments? Please check us out on IRC via irc://udderweb.com/#uw
