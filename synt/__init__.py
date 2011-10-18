# -*- coding: utf-8 -*-
from synt.trainer import train
from synt.collector import collect, fetch
from synt.guesser import Guesser
from synt.tester import test

try:
    import argparse
except ImportError:
    raise

VERSION = '0.1.0'

def main():

    parser = argparse.ArgumentParser(description='Tool to interface with synt, provides a way to train, collect and guess from the command line.')

    subparsers = parser.add_subparsers(dest='parser')

    #Train Parser
    train_parser = subparsers.add_parser(
            'train', 
            help='Train a classifier.'
    )
    train_parser.add_argument(
        'samples',
        type=int,
        help="The amount of samples to train on. Uses the samples.db",
    )
    train_parser.add_argument(
        '--classifier',
        default='naivebayes',
        choices=('naivebayes',),
        help="The classifier to use. We currently only support naivebayes.",
    )
    train_parser.add_argument(
        '--best_features',
        type=int,
        default=5000,
        help="The amount of best words to use, or best features. By default 5000 are used.",
    )
    train_parser.add_argument(
        '--purge',
        default='no',
        choices=('yes', 'no'),
        help="Yes to purge the redis database. By default no."
    ) 
    train_parser.add_argument(
        '--processes',
        default=4,
        help="Will utilize multiprocessing if available with this number of processes. By default 4."
    )

    #Collect parser
    collect_parser = subparsers.add_parser(
            'collect',
            help='Collect samples.'
    )
    collect_parser.add_argument(
        '--commit_every',
        default=200,
        type=int,
        help="Write to sqlite database after every 'this number'. Default is 200",
    )
    collect_parser.add_argument(
        '--max_collect',
        default=2000000,
        type=int,
        help="The amount to stop collecting at. Default is 2 million",
    )

    #Fetch parser
    fetch_parser = subparsers.add_parser(
            'fetch', 
            help='Fetches premade sample database.'
    )
    fetch_parser.add_argument(
            'fetch', 
            nargs='?',
            default=True,
            help="Fetches the default samples database from github."
    )

    #Guess parser
    guess_parser = subparsers.add_parser(
            'guess',
            help='Guess sentiment'
    )
    guess_parser.add_argument(
        'guess', 
        nargs='?',
        default=True,
        help="Starts the guess prompt.",
    )
   
    #Tester parser
    tester_parser = subparsers.add_parser(
            'tester', 
            help='Test accuracy of classifier.',
    )
    tester_parser.add_argument(
        'samples', 
        type=int,
        help="The amount of samples to test on."
    )
    tester_parser.add_argument(
        '--classifier',
        default='naivebayes',
        choices=('naivebayes',),
        help="The classifier to use. Currently we only support naivebayes."
    )
    tester_parser.add_argument(
        '--neutral_range',
        default=0.0,
        type=float,
        help="Neutral range to use. By default there isn't one."
    )
   
    args = parser.parse_args()

    if args.parser == 'train':
        
        purge = False
        if not args.purge == 'no':
            purge = True

        train(
            samples       = args.samples,
            classifier    = args.classifier,
            best_features = args.best_features,
            processes     = args.processes,
            pruge         = purge,
        )

    elif args.parser == 'collect':
        collect(
            commit_every = args.commit_every,
            max_collect  = args.max_collect,
        )    
    
    elif args.parser == 'fetch':
        fetch()

    elif args.parser == 'guess':
        g = Guesser()
    
        print("Enter something to calculate the synt of it!")
        print("Just press enter to quit.")
    
        while True:
            text = raw_input("synt> ")
            if not text:
                break    
            print('Guessed: {}'.format(g.guess(text)))
    
    elif args.parser == 'tester':
        test(
            test_samples  = args.samples,
            classifier    = args.classifier,
            neutral_range = args.neutral_range,
        )
        

if __name__ == '__main__':
    main()
