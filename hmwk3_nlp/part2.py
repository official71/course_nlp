import collections

import os
import argparse
import random
import subprocess

import utils
import graphParser

def main(train_file, test_file, output, model, num_epochs, embeddings_init=None, pos_d=0, seed=0):
    vocab = utils.Vocabulary(train_file)
    print('reading train...')
    train = list(utils.read_conll(train_file))
    print('read {} examples'.format(len(train)))
    print('reading test...')    
    test = list(utils.read_conll(test_file))
    print('read {} examples'.format(len(test)))
    
    print 'Initializing lstm parser:'
    parser = graphParser.GraphParser(vocab.num_words,
                                     vocab.num_rel,
                                     pos_d=pos_d,
                                     pos_V=vocab.num_pos if pos_d else None,
                                     embeddings_init=embeddings_init,
                                     seed=seed,
                                     verbose=True)

    print('formatting test data...')
    test_indices, test_pos_indices, test_arcs, test_labels = vocab.process(test,
                                                                           deterministic=True)

    for epoch in range(num_epochs):
        print 'Starting epoch', epoch
        loss = 0

        #shuffle the training data
        random.shuffle(train)

        #convert to indices, sample, etc
        indices, pos_indices, gold_arcs, gold_labels = vocab.process(train)
        #train and return loss
        loss = parser.train(indices, gold_arcs, gold_labels, pos_indices if pos_d else None)
        
        #get predicted labels for test set
        predicted_arcs, predicted_labels = parser.predict(test_indices, test_pos_indices if pos_d else None)

        #write the predictions to a CONLL formatted file
        devpath = os.path.join(output, 'dev_tmp.conll')
        utils.write_conll(devpath, vocab.entry(test_indices, test_pos_indices, predicted_arcs, predicted_labels))

        #call the CONLL evaluation script and extract the LAS and UAS        
        p = subprocess.Popen(['perl', 'src/utils/eval.pl', '-g', test_file,  '-s', devpath],
                             stdout = subprocess.PIPE)
        out, err = p.communicate()
        las = float(out.splitlines()[0].split()[-2])
        uas = float(out.splitlines()[1].split()[-2])

        #do whatever metrics
        utils.metrics(loss, uas, las)

        #save the current model
        parser.save(os.path.join(output, os.path.basename(model)), vocab.idx2word, vocab.idx2pos if pos_d else None)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    parser.add_argument('output')
    parser.add_argument('model')
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--embeddings_init')
    parser.add_argument('--pos_d', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dynet-mem')
    
    args = vars(parser.parse_args())
    args.pop('dynet_mem')
    
    main(**args)
    

