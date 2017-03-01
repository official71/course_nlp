from providedcode import dataset
from providedcode.dependencygraph import DependencyGraph
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor, set_feature_option
from transition import Transition
from re import match
import sys
import time

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print 'Wrong input arguments'
        exit(1)

    arg_lan = sys.argv[1].lower()
    if arg_lan in ['english', 'en', 'eng']:
        language = 'english'
    elif arg_lan in ['swedish', 'sw', 'swe']:
        language = 'swedish'
    else:
        print 'Invalid arguments: ' + arg_lan
        exit(1)

    arg_fo = sys.argv[2].lower()
    feature_options = sys.argv[2].lower().split('-')
    for fo in feature_options:
        if not match(r's0|s1|b0|b1|b2|b3[f|t|c]', fo):
            print "Invalid argument: " + fo
            exit(1)
    set_feature_option(feature_options)


    if language is 'swedish':
        traindata = dataset.get_swedish_train_corpus().parsed_sents()
    else:
        traindata = dataset.get_english_train_corpus().parsed_sents()


    try:
        time.clock()
        tp = TransitionParser(Transition, FeatureExtractor)
        tp.train(traindata)

        fname = language + '.' + arg_fo
        tp.save(fname + '.model')
        # tp.save('swedish.model')

        if language is 'swedish':
            labeleddata = dataset.get_swedish_dev_corpus().parsed_sents()
            blinddata = dataset.get_swedish_dev_blind_corpus().parsed_sents()
        else:
            labeleddata = dataset.get_english_dev_corpus().parsed_sents()
            blinddata = dataset.get_english_dev_blind_corpus().parsed_sents()
        
        # tp = TransitionParser.load('badfeatures.model')

        parsed = tp.parse(blinddata)

        with open(fname + '.conll', 'w') as f:
            for p in parsed:
                f.write(p.to_conll(10).encode('utf-8'))
                f.write('\n')

        ev = DependencyEvaluator(labeleddata, parsed)
        print "\n-----------------------------------------------"
        print "language: " + language + "\tfeatures: " + arg_fo
        print "time: " + str(time.clock()) + ' sec'
        print "UAS: {} \nLAS: {}".format(*ev.eval())

        # parsing arbitrary sentences (english):
        # sentence = DependencyGraph.from_sentence('Hi, this is a test')
        # sentence = DependencyGraph.from_sentence('The team eliminated the crisis')

        # tp = TransitionParser.load('english.model')
        # parsed = tp.parse([sentence])
        # print parsed[0].to_conll(10).encode('utf-8')
    except NotImplementedError:
        print """
        This file is currently broken! We removed the implementation of Transition
        (in transition.py), which tells the transitionparser how to go from one
        Configuration to another Configuration. This is an essential part of the
        arc-eager dependency parsing algorithm, so you should probably fix that :)

        The algorithm is described in great detail here:
            http://aclweb.org/anthology//C/C12/C12-1059.pdf

        We also haven't actually implemented most of the features for for the
        support vector machine (in featureextractor.py), so as you might expect the
        evaluator is going to give you somewhat bad results...

        Your output should look something like this:

            UAS: 0.23023302131
            LAS: 0.125273849831

        Not this:

            Traceback (most recent call last):
                File "test.py", line 41, in <module>
                    ...
                    NotImplementedError: Please implement shift!


        """
