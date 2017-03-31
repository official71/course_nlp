import random
from collections import Counter
import re

import numpy as np
import math
from random import random

#tokens for unknown words and numbers
UNKNOWN = '__UNK__'
NUMBER = '__NUM__'

#IMPLEMENT THE NORMALIZE METHOD HERE
# is string s a numeric representation?
# inspired by the answers in:
# http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float-in-python
def is_numeric(s):
    try:
        if not math.isnan(float(s)):
            return True
        return False
    except ValueError:
        return False

def normalize(word):
    '''
    takes in a word string and returns the NUMBER token if a number and lowercase otherwise
    '''
    # raise NotImplementedError
    if is_numeric(word):
        return NUMBER
    return word.lower()

class ConllEntry:
    '''
    class for representing CONLL formatted dependency data
    '''
    def __init__(self, id, form, pos, cpos, parent_id=None, relation=None):
        '''
        initialization for ConllEntry class
        id - the index of the item (first column in CONLL file)
        form - word
        pos - fine-grained POS
        cos - coarse-grained POS
        parent_id - optional parent index
        relation - optional dependency relation label
        '''
        
        self.id = id
        self.form = form
        
        #the normalized representation of the word (use this one for modeling)
        self.norm = normalize(form)

        #standardize the POS tags by uppercasing them
        self.cpos = cpos.upper()
        self.pos = pos.upper()

        #the true parent and relation
        self.parent_id = parent_id
        self.relation = relation

        #the predicted parent and relation
        self.pred_parent_id = None
        self.pred_relation = None

class Vocabulary:
    '''
    a class for converting between indices and tokens
    '''
    def __init__(self, conll_path):
        '''
        initialization for Vocabulary class
        
        conll_path - the full file path to the training data
        '''

        #frequency counts for each of words, POS, and dependency relations        
        self.words = Counter()
        self.pos = Counter()
        self.rel = Counter()

        #go through each training example and add them to the counter
        for i,sentence in enumerate(read_conll(conll_path)):
            for j,node in enumerate(sentence):
                self.words.update([node.norm])
                self.pos.update([node.pos])
                self.rel.update([node.relation])

        #lookup table of indices to strings
        self.idx2word = self.words.keys()
        self.idx2pos = self.pos.keys()
        self.idx2rel = self.rel.keys()

        #need to add the unknown token
        self.idx2word = [UNKNOWN] + self.idx2word

        #reverse lookup table for strings to indices
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.pos2idx = {w: i for i, w in enumerate(self.idx2pos)}
        self.rel2idx = {w: i for i, w in enumerate(self.idx2rel)}

    @property
    def num_words(self):
        return len(self.idx2word)

    @property
    def num_pos(self):
        return len(self.idx2pos)

    @property
    def num_rel(self):
        return len(self.idx2rel)
        
    def process(self, data, deterministic=False):
        '''
        convert a list of ConllEntry to a list of indices for each token type
        
        data - a list of lists of ConllEntrys
        deterministic - a parameter indicating whether to randomly replace words with the UNKNOWN token
        
        returns indices, pos indices, parent indices, and dependency relation labels
        '''

        #YOUR IMPLEMENTATION GOES HERE
        # raise NotImplementedError
        indices, pos_indices, arcs, labels = [], [], [], []
        for se in data:
            i, p, a, l = [], [], [], []
            for w in se:
                # indices
                idx = self.word2idx.get(w.norm)
                if idx is None:
                    i.append(self.word2idx[UNKNOWN])
                elif not deterministic and random() < 0.25 / (0.25 + self.words[w.norm]):
                    # random sampling
                    # print "random sampling occurred for " + w.norm + " from " + str(idx)
                    i.append(self.word2idx[UNKNOWN])
                else:
                    # don't do sampling
                    i.append(idx)

                # pos_indices
                pos = self.pos2idx.get(w.pos)
                if pos is None:
                    print "Should not happen: unseen POS " + w.pos
                    p.append(-1)
                else:
                    p.append(pos)

                # arcs
                a.append(w.parent_id)

                # labels
                label = self.rel2idx.get(w.relation)
                if label is None:
                    print "Should not happen: unseen rel " + w.relation
                    l.append(-1)
                else:
                    l.append(label)

            indices.append(i)
            pos_indices.append(p)
            arcs.append(a)
            labels.append(l)
        return indices, pos_indices, arcs, labels

        
    def entry(self, indices, pos_indices, arcs, labels):
        '''
        generator to convert numeric indices to ConllEntry format
        
        indices - indices for words
        pos_indices - indices for POS
        arcs - arcs (parent indices into the sentence)
        labels - dependency relation indices
        
        yields a list of ConllEntry tokens
        '''
        
        for i in range(len(indices)):
            #root is the 0th entry
            root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', -1, 'rroot')
            tokens = [root]
            
            for j in range(1, len(indices[i])):
                #convert indices back to some token
                tokens.append(ConllEntry(j,
                                         self.idx2word[indices[i][j]],
                                         self.idx2pos[pos_indices[i][j]],
                                         '-',
                                         arcs[i][j],
                                         self.idx2rel[labels[i][j]]))
                tokens[-1].pred_parent_id = arcs[i][j]
                tokens[-1].pred_relation = self.idx2rel[labels[i][j]]
            yield tokens
                
def read_conll(fn, min_length = 1, max_length = 100):
    '''
    read one line at a time from a CONLL format file, returning whole sentences
    
    fn - full path filename for CONLL data
    min_length - optional
    max_length - optional
    
    yields a list of ConllEntry tokens
    '''

    #root is always 0th entry
    root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', -1, 'rroot')
    tokens = [root]
    
    with open(fn) as fh:
        #iterate over each line in the file
        for line in fh:
            tok = line.strip().split()

            if not tok:
                #if we encounter a whitespace line, the data point has ended
                if len(tokens)>min_length and len(tokens)<max_length :
                    yield tokens
                tokens = [root]
            else:
                #add the ID (0), word (1), fine-grained pos (4), coarse pos (3), parent id (6), and relation (7)
                tokens.append(ConllEntry(int(tok[0]),
                                         tok[1],
                                         tok[4],
                                         tok[3],
                                         int(tok[6]) if tok[6] != '_' else -1,
                                         tok[7]))
                
        if len(tokens) > min_length and len(tokens) < max_length:
            yield tokens


def write_conll(fn, conll_gen):
    '''
    write the data to a CONLL formatted file

    fn - full path filename
    conll_gen - iterator where each item is a list of CONLL tokens
    '''
    
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write('\t'.join([str(entry.id),
                                    entry.form,
                                    '_',
                                    entry.cpos,
                                    entry.pos,
                                    '_',
                                    str(entry.pred_parent_id),
                                    entry.pred_relation,
                                    '_',
                                    '_']))
                fh.write('\n')
            fh.write('\n')

def metrics(loss, uas, las):
    '''
    loss - training loss
    uas - unlabeled accuracy
    las - labeled accuracy
    '''

    #YOU MAY WANT TO MODIFY THIS CODE HERE
    print('Training loss: {}'.format(loss))
    print('UAS: {}'.format(uas))
    print('LAS: {}'.format(las))
