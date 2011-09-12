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

## Usage / Installation ##

  1. Grab the latest source:
    
    ```git clone git@github.com:Tawlk/synt.git```


  2. Grab the sample database to train on (or build one (below)):

    ```wget -O - "https://github.com/downloads/Tawlk/synt/sample_data.bz2" | bzcat | sqlite3 sample_data.db```

    If you'd prefer to build a fresh sample db and have the time:

    ```python
    python2.7 collector.py
    ```
    
    Ctrl+C to stop it. 

    **Note**: This also can update an existing database, if you decided to
    initially wget the db but want to add more content the above should still
    work.

  3. Run ```python2.7 trainer.py``` to build classifier.

    Note this requires Redis to be started!


  4. Usage:

    ```python
    from synt import guess
    guess('My mother in law makes me do bad things and I think pandas are icky!')
    ```
    Or on the command line:

    ```python -c "from synt import guess;guess('I want to chase poodles. That makes me happy any joyful :-D')"```

    This will return a score from -1 .. 1 negative to positive respectivley.
    Anything closer 0 should be considered neutral.


## Notes ##
    
  Use at your own risk. You may be eaten by a grue.

  Questions/Comments? Please check us out on IRC via irc://udderweb.com/#uw
