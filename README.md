# Synt #

  <http://github.com/Tawlk/synt>

## About ##

  Synt (pronounced: "cent") is a python library for sentiment
  classification on social text.

  The end-goal is to have a simple library that "just works". It should
  have an easy barrier to entry and be thoroughly documented.


## Current Features ##

  * Can collect negative/positive tweets from twitter and store it to a local
    database (can also fetch a pre-existing samples database)
  * Can train a classifier based on a samples database
  * Can classifiy text and output a score between -1 and 1. (where -1 is
    negative, +1 is positive and anything close to 0 can be considered neutral)
  * abilitiy to collect, train, guess, and test (accuracy) from cli


## Requirements ##

  * A running [Redis](http://redis.io) server
  * [pip](http://www.pip-installer.org/en/latest/index.html)
  * [virtualenv](http://www.virtualenv.org/en/latest/index.html) (recommended)
  * [python2.7](http://www.python.org/getit/releases/2.7/) (no support for
    python3.x)
    * sqlite3 issue was fixed in 2.7 [issue](http://code.google.com/p/pysqlite/source/detail?r=9e3fa82223b89ca4e7f9eadedc1297ab5c3eebd9)
    * argparse
  * [PyYAML](http://pyyaml.org/) (pip install pyyaml)
    * unfortnatley nltk requires pyyaml before it can be installed [bug](http://code.google.com/p/nltk/issues/detail?id=508)


## Usage / Installation ##

**Note: Many of these commands have additional arguments you can pass, use
the -h flag to get help on any particular command and see more options.**

  1. Grab the latest synt:

    ```bash
    pip install -e git+https://github.com/Tawlk/synt/#egg=synt
    ```

  2. Grab the sample database to train on (or build one (below)):

    **Note: On your first run of any cli command a config will be copied into
    ~/.synt/config.py that you should configure. It uses sane defaults. This
    will only happen on the first run of synt.**

    ```bash
    synt fetch --db_name "mysamples.db"
    ```

    By default it will be stored as 'samples.db'.

    If you'd prefer to build a fresh sample db and have the time, just run collect with
    the desired amount.

    ```bash
    synt collect --max_collect 10000 --db_name 'awesome.db'
    ```

    **Note:** You can also increment samples in a database by providing the
    same db name.


  3. Train classifier

    A basic example of training

    ```bash
    synt train 'samples.db' 20000
    ```

    Train takes two required arguments: a training database (name), and the amount of
    samples to train on.


  4. Classifier accuracy

    At this point you might want to see the classifiers accuracy on the
    training data.

    ```bash
    synt accuracy
    ```

    Accuracy takes a number of testing samples. By default 25% of your training
    sample count will be used as the testing set. You can over-ride this by
    providing the --test_samples argument.

    The database used for these testing samples will be the same as the database
    used to train. The testing samples will be new samples and can be
    guaranteed to be samples the classifier hasn't already seen.


  5. Guessing/classifying text

    You should now have a trained classifier and its time to see
    some classification of text.

    ```bash
    synt guess
    ```

    This will drop you into a synt prompt where you can write text and see
    the score between -1 and 1.

    You can alternativley also just classify text without having to drop into
    a prompt:

    ```bash
    synt guess --text "i like ponies and rainbows"
    ```


## Notes ##

  * We have acheived best accuracy using stopwords filtering with tweets collected on
    negwords.txt and poswords.txt (see downloads).

  * In the future we will also add the MaxEnt and Decision tree classifiers and
    the functionality to do clasiffier voting.

  * Note that this is optimized for classification on social text as this is our
    primary usecase. However, with a little tweaking it should be possible to
    get good results on other corpus'.

  This code is still in production; use at your own risk. You may be eaten by a grue.

  Questions/Comments? Please check us out on IRC via irc://irc.freenode.net/#tawlk
