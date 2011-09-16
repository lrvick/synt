from synt.trainer import train
from synt.collector import collect
from synt.guesser import guess
from synt.tester import test

import argparse

def main():

    parser = argparse.ArgumentParser(description='Tool to interface with synt, provides a way to train, collect and guess from the command line.')

    subparsers = parser.add_subparsers(help='sub-command help', dest='parser')

    #train command
    parser_train = subparsers.add_parser('train', help='Train a classifer')
    parser_train.add_argument(
            '--train_samples',
            action='store',
            type=int,
            default=2000,
            help="""The amount of samples to train on."""
    )
    
    parser_train.add_argument(
            '--wc_samples',
            action='store',
            type=int,
            default=2000,
            help="""We store a word:count mapping to determine a list of useful and popular words to use.
            This is the the number of samples to generate our words from. Generally you want this number to
            be pretty high as it will gradually reduce variations and produce a consistent set of useful words."""
    )

    parser_train.add_argument(
            '--wc_range',
            action='store',
            type=int,
            default=2000,
            help="""This is the actual amount of words to use to build freqDists. By this point (depending on how many word samples used) you will have a lot of tokens. Most of these tokens are uninformative and produce nothing but noise. This is the first layer of cutting down that batch to something reasonable. The number provided will use words
            from 0 .. wc_range. Words are already sorted by most frequent to least."""
            
    )
    parser_train.add_argument(
        '--fresh',
        action='store',
        type=int,
        default=False,
        help="""If True this will force a new train, useful to test various sample, wordcount combinations. 1 = True 0 = False"""
    )
    
    parser_train.add_argument(
        '--verbose',
        action='store',
        type=int,
        default=True,
        help="""Displays log info to stdout by default. 1 = True 0 = False"""
    )

    #collect command
    parser_collect = subparsers.add_parser('collect', help='Collect sample data.')
    parser_collect.add_argument('fetch', help='Grab the sample_database')
    parser_collect.add_argument('--time', action='store', type=int, default=500)

    #guess command
    parser_guess = subparsers.add_parser(
        'guess',
        description="Guess' sentiment. This relies on a trained classifier to exist in the database which means you should run 'train' before attempting to guess. The output is a float between -1 and 1 detailing how negative or positive the sentiment is. Anything close to 0 should be treated as relativley neutral.",
    )
    parser_guess.add_argument(
            'text',
            action='store',
            help = 'Text to guess on.'
    )

    #tester commmand
    parser_tester = subparsers.add_parser(
        'tester',
        description = """Tests the accuracy of the classifier."""
    )
    
    parser_tester.add_argument(
        '--test_samples',
        action='store',
        type=int,
        help='The amount of test_samples to test against.'
    )

    args = parser.parse_args()
    
    if args.parser == 'train':
        train(
            train_samples=args.train_samples,
            wordcount_samples=args.wc_samples,
            wordcount_range=args.wc_range,
            verbose=args.verbose,
            force_update=args.fresh,
    )
   
    if args.parser == 'collect':
        #interface susceptible to change
        pass

    if args.parser == 'guess':
        text = args.text.strip()
        print(guess(text=u''+text))

    if args.parser == 'tester':
        test(test_samples=args.test_samples)

if __name__ == '__main__':
    main()
