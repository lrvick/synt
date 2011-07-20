# Synt #
  
  <http://github.com/lrvick/synt>

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

  1. Collect a large amount of samples using the automated trainer
    
        python auto-trainer.py

    As a rule of thumb anything less than 200,000 leads to less that useful results.

    Optionally you can use the rather large ( 2mil tweets / 100M ) pre-collected database 
    found in the downloads section on github

        wget -O - "https://github.com/downloads/Tawlk/synt/sample_data.bz2" | bzcat | sqlite3 sample_data.db
        

  2. From here, you can now begin to start evaluating text.

    In a a python script:

        import synt
        synt.guess('My mother in law makes me do bad things and I think pandas are icky!')

    Or on the command line:

        python -c "import synt; synt.guess('I want to chase poodles. That makes me happy any joyful :-D')"

    The first time you do a guess, it will generate and save a classifier blob based on
    200k samples, (100K neg, 100k pos). This default will get you around 80% accuracy.

    You can adjust this, (by changing a couple hard-coded variables) but going very far 
    beyond 200k at this time will probably make your computer cry and say mean things 
    to you, before passing out.
  
## Notes ##
    
  Use at your own risk. You may be eaten by a grue.

  Questions/Comments? Please check us out on IRC via irc://udderweb.com/#uw
