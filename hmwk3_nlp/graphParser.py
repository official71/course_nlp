import time
import json

import numpy as np

import dynet

from layers import MLP, LSTM
from decoder import parse_proj

class GraphParser:
    '''
    class for training and predicting dependency arcs and labels using a graph-based algorithm
    '''
    def __init__(self, word_V, dep_V,
                 word_d=100, pos_d=25, mlp_d=100, mlp_label_d=100,
                 num_lstm_layers=2, lstm_d=125, embeddings_init=None, pos_V=None,
                 seed=0, verbose=False):
        '''
        word_V - size of word vocab
        dep_V - size of relation label vocab
        word_d - dimension of word embeddings
        pos_d - dimension of POS embeddings
        mlp_d - dimension of hidden layer for arc prediction MLP
        mlp_label_d - dimension of hidden layer for label prediction MLP
        num_lstm_layers - number of bi-directional LSTM layers to stack
        lstm_d - dimension of hidden state in the LSTM
        embeddings_init - use pre-trained embeddings
        pos_V - size of POS vocab
        seed - random seed for initialization
        verbose - whether to print information about these parameters
        '''
        
        if verbose:
            print('Word vocabulary size: {}'.format(word_V))
            print('Dependency relation vocabulary size: {}'.format(dep_V))
            print('POS vocabulary size: {}'.format(pos_V))        
        
        self.word_V = word_V
        self.dep_V = dep_V
        self.pos_V = pos_V
        
        self.word_d=word_d
        self.pos_d=pos_d
        self.mlp_d=mlp_d
        self.mlp_label_d=mlp_label_d
        self.lstm_layers=num_lstm_layers
        self.lstm_d=lstm_d

        np.random.seed(seed)
        
        self.model = dynet.Model()

        #embedding layers for words and POS        
        self.embeddings =  self.model.add_lookup_parameters((self.word_V, self.word_d))
        if pos_V is not None:
            self.pos_embeddings = self.model.add_lookup_parameters((self.pos_V, self.pos_d))

        #bi-directional LSTM layers
        #embeddings -> layer1 -> layer2
        lstm_layers = []        
        for i in range(num_lstm_layers):
            input_d = word_d
            if i:
                input_d = 2*lstm_d
            elif pos_V is not None:
                input_d += pos_d
            
            fwd_lstm_layer = LSTM(self.model, input_d, lstm_d)
            rev_lstm_layer = LSTM(self.model, input_d, lstm_d, reverse=True)
            lstm_layers.append((fwd_lstm_layer, rev_lstm_layer))

        #arc prediction MLP
        #layer2(i), layer2(j) -> concatenate -> score
        mlp_layer = MLP(self.model, lstm_d*4, mlp_d, 1)
        #label prediction MLP
        if mlp_label_d:
            mlp_label_layer = MLP(self.model, lstm_d*4, mlp_label_d, dep_V)            
        else:
            mlp_label_layer = None

        #train the model using Adam optimizer
        self.trainer = dynet.AdamTrainer(self.model)

        #take in word and pos_indices, return the output of the 2nd layer                    
        def get_lstm_output(indices, pos_indices=None):
            embeddings_out = [self.embeddings[w] for w in indices]
            x = embeddings_out
            
            if pos_V is not None and pos_indices is not None:
                x = []
                for i,input in enumerate(embeddings_out):
                    x.append(dynet.concatenate([input, self.pos_embeddings[pos_indices[i]]]))

            for i in range(num_lstm_layers):
                x_1 = lstm_layers[i][0].get_output(x)[0]
                x_2 = lstm_layers[i][1].get_output(x)[0]
                x = [dynet.concatenate([x_1[i], x_2[i]]) for i in range(len(indices))]
    
            return x

        self.states = get_lstm_output

        #score all arcs from i to j using the arc prediction MLP
        def score_arcs(states, value=True):
            length = len(states)
            scores = [[None for i in range(length)] for j in range(length)]
            
            for i in range(length):
                for j in range(length):
                    score = mlp_layer.get_output(dynet.concatenate([states[i], states[j]]))
                    if value:
                        scores[i][j] = score.scalar_value()
                    else:
                        scores[i][j] = score

            return scores
        
        self.score_arcs = score_arcs

        #score all labels at i using the label prediction MLP
        def score_labels(states, arcs, value=True):
            scores = []
                                    
            for i in range(len(states)):
                score = mlp_label_layer.get_output(dynet.concatenate([states[i],
                                                                      states[arcs[i]]]))
                if value:
                    scores.append(score.value())
                else:
                    scores.append(score)

            return scores

        self.score_labels = score_labels

    #loss function for arcs (use a margin loss for invalid arcs in the current best parse)
    def arc_loss(self, gold_arcs, arc_scores):
        errors = []
        arc_scores_values = np.array([[j.value() for j in i] for i in arc_scores])
        arcs = parse_proj(arc_scores_values, gold_arcs)
        
        for i in range(len(gold_arcs)):
            if gold_arcs[i] != arcs[i]:
                error = arc_scores[arcs[i]][i] - arc_scores[gold_arcs[i]][i]
                errors.append(error)

        return errors

    #loss function for labels (use a margin loss for the gold label and highest scoring incorrect label)
    def label_loss(self, gold_labels, label_scores):
        errors = []
        label_scores_values = [i.value() for i in label_scores]

        for dependent in range(1, len(gold_labels)):
            sorted_label_scores = np.argsort(label_scores_values[dependent])
            wrong_label = sorted_label_scores[-1]
            if wrong_label == gold_labels[dependent]:
                wrong_label = sorted_label_scores[-2]
            gold_label = gold_labels[dependent]
            if label_scores_values[dependent][gold_label] < label_scores_values[dependent][wrong_label] + 1:
                errors.append(label_scores[dependent][wrong_label] - label_scores[dependent][gold_label])

        return errors

    #given a list of indices, score all arcs and return the highest scoring dependency tree
    def parse(self, indices, arcs=None, pos_indices=None):
        states = self.states(indices, pos_indices)
        scores = np.array(self.score_arcs(states))

        return parse_proj(scores, arcs)

    #given a list of indices and arcs, return the highest scoring labels
    def label(self, indices, arcs, pos_indices=None):
        states = self.states(indices, pos_indices)
        scores = np.array(self.score_labels(states, arcs))

        return np.argmax(scores, axis=-1)

    #train the model given a list of list of indices and gold data
    def train(self, indices, gold_arcs, gold_labels, pos_indices=None):
        total_arc_loss = 0
        total_label_loss = 0
        start = time.time()
        
        for i in range(len(indices)):
            states = self.states(indices[i], pos_indices[i] if pos_indices is not None else None)
            arc_scores = self.score_arcs(states, value=False)
            label_scores = self.score_labels(states, gold_arcs[i], value=False)
            
            arc_loss = self.arc_loss(gold_arcs[i], arc_scores)
            label_loss = self.label_loss(gold_labels[i], label_scores)

            if len(arc_loss) > 0:
                arc_loss = dynet.esum(arc_loss)
            else:
                arc_loss = dynet.scalarInput(0)
            if len(label_loss) > 0:
                label_loss = dynet.esum(label_loss)
            else:
                label_loss = dynet.scalarInput(0)

            loss = dynet.esum([arc_loss, label_loss])
            arc_loss = arc_loss.value()
            label_loss = label_loss.value()
            total_arc_loss += arc_loss
            total_label_loss += label_loss
            loss.backward()
            self.trainer.update()
            
            dynet.renew_cg()
        print(time.time()-start)    
        return total_arc_loss, total_label_loss

    #parse a sentence given indices, return the highest scoring arcs and labels
    def predict(self, indices, pos_indices=None):
        all_arcs = []
        all_labels = []
        for i in range(len(indices)):
            arcs = self.parse(indices[i], None, pos_indices[i] if pos_indices is not None else None)
            labels = self.label(indices[i], arcs, pos_indices[i] if pos_indices is not None else None)

            all_arcs.append(arcs)
            all_labels.append(labels)
            dynet.renew_cg()
            
        return all_arcs, all_labels

    #save the model, and optionally the word and POS embeddings
    def save(self, filename, idx2word=None, idx2pos=None):
        self.model.save(filename)

        x2embedding = {}
        if idx2word:
            word2embedding = {}
            for i in range(self.word_V):
                word2embedding[idx2word[i]] = self.embeddings[i].value()
            x2embedding['word'] = word2embedding
            
        if idx2pos:
            pos2embedding = {}
            for i in range(self.pos_V):
                pos2embedding[idx2pos[i]] = self.pos_embeddings[i].value()
            x2embedding['pos'] = pos2embedding

        if len(x2embedding):
            with open(filename + '_embeddings.json', 'w') as f:
                json.dump(x2embedding, f)
