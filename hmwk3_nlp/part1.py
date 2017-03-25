import random
from collections import defaultdict
from itertools import count
import sys

from dynet import *
import numpy as np

from layers import MLP, SimpleRNN, LSTM

#dimensionality of character embeddings
INPUT_DIM = 50
#dimensionality of RNN hidden state
HIDDEN_DIM = 50

characters = list("abcdefghijklmnopqrstuvwxyz ")
characters.append("<EOS>")

#mappings between characters and their index in the embedding matrix
int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

rnn_types = {'simple_rnn': SimpleRNN,
             'lstm': LSTM}

def main(rnn_type = 'simple_rnn'):
    '''
    train a language model on character data
    '''
    
    assert(rnn_type in ('simple_rnn', 'lstm'))
    
    model = Model()

    lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))        
    rnn = rnn_types[rnn_type](model, INPUT_DIM, HIDDEN_DIM)
    mlp = MLP(model, HIDDEN_DIM, HIDDEN_DIM, VOCAB_SIZE,
              output_nonlinearity='softmax', num_layers=0)

    #our single training example    
    sentence = "a quick brown fox jumped over the lazy dog"

    train(model, rnn, mlp, lookup, sentence)
    
def do_one_sentence(rnn, mlp, lookup, sentence):
    '''
    return compute loss of RNN for one sentence
    '''
    
    #new computation graph every time
    renew_cg()
    
    # pad the sentence    
    sentence = ["<EOS>"] + list(sentence) + ["<EOS>"]

    #convert characters to indices
    sentence = [char2int[c] for c in sentence]

    #get the embedding and hidden state for every character in the sentence
    states = rnn.get_output([lookup[i] for i in sentence])
    
    loss = []
    for index,char in enumerate(sentence[:-1]):
        #get the softmax output of the MLP, and calculate the loss using the gold output
        probs = mlp.get_output(states[0][index])
        loss.append( -log(probs[sentence[index+1]]))
        
    loss = esum(loss)
    return loss        

def generate(rnn, mlp, lookup, sample=False):
    '''
    generate text from the model one character at a time
    '''
    def sample(probs):
        rnd = random.random()
        for i,p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    # need a new computation graph every time
    renew_cg()

    #get the first state starting with the pad character
    state = rnn.get_output([lookup[char2int["<EOS>"]]])
    out=[]
    while True:
        #get the probability vector for the next character
        probs = mlp.get_output(state[0][0])
        probs = probs.vec_value()

        if sample:
            next_char = sample(probs)
        else:
            #we should be doing a beam search here but instead we are
            #just taking the max probability character from the current state
            next_char = np.argmax(probs)

        out.append(int2char[next_char])
        
        if out[-1] == "<EOS>" or len(out) > 100: break

        #we generate the next character, but start from the previous hidden state
        state = rnn.get_output([lookup[next_char]], *(i[0] for i in state))
        
    return "".join(out[:-1]) # strip the <EOS>

# train, and generate every 5 samples
def train(model, rnn, mlp, lookup, sentence):
    #train using SGD
    trainer = SimpleSGDTrainer(model)
    
    for i in xrange(200):
        #calculate the loss and backpropagate, updating the params
        loss = do_one_sentence(rnn, mlp, lookup, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        
        if i % 5 == 0:
            print loss_value,
            print generate(rnn, mlp, lookup)
        
if __name__ == '__main__':
    main(sys.argv[1])
