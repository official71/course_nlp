import numpy as np

import dynet
from dynet import parameter
from dynet import cmult, logistic, dot_product, tanh, softmax, rectify

#dictionary of useful nonlinearities for your convenience
nonlinearities = {'tanh': tanh,
                  'sigmoid': logistic,
                  'logistic': logistic,
                  'rectifier': rectify,
                  'rectify': rectify,
                  'relu': rectify,
                  'softmax': softmax,
                  None: lambda x:x}

class MLP:
    '''
    Multi-Layer Perceptron class (for any number of hidden layers >= 0)
    '''
    def __init__(self, model, d, hd, num_classes,
                 hidden_nonlinearity='tanh', output_nonlinearity=None,
                 num_layers = 1):
        '''
        init method for MLP class
        model - pre-instantiated dynet Model object
        d - dimensionality of input layer
        hd - dimensionality of hidden layer
        num_classes - dimensionality of output layer
        hidden_nonlinearity - nonlinearity for all hidden layers, must be valid in nonlinearities
        output_nonlinearity - nonlinearity for output layer, must be valid in nonlinearities
        '''
        
        self.d = d
        self.hd = hd
        self.num_classes = num_classes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.num_layers = num_layers

        #parameters for output layer
        #weight matrix is projection of hidden layer onto output layer
        self.W_o = model.add_parameters((num_classes, hd))
        #bias for each output class
        self.b_o = model.add_parameters((num_classes,))
        self.weights = []
        self.biases = []

        #each hidden layer maps the previous layer to the next layer
        in_d = d
        for _ in range(self.num_layers):
            self.weights.append(model.add_parameters((hd, in_d)))
            self.biases.append(model.add_parameters((hd, )))
            in_d = hd                                
            
        self._params = self.weights + self.biases + [self.W_o, self.b_o]

    @property
    def params(self):
        return self._params
        
    def get_output(self, input):
        '''
        apply the mlp to an input vector

        input - a vector of dimensionality d
        '''
        
        weights = [parameter(i) for i in self.weights]
        biases = [parameter(i) for i in self.biases]
        W_o = parameter(self.W_o)
        b_o = parameter(self.b_o)

        #IMPLEMENT YOUR CODE BELOW
        # raise NotImplementedError
        x = input
        for w, b in zip(weights, biases):
            output = nonlinearities[self.hidden_nonlinearity](w * x + b)
            x = output
        return nonlinearities[self.output_nonlinearity](W_o * x + b_o)
        
class SimpleRNN:
    '''
    Simple Recurrent Neural Network class
    '''
    def __init__(self, model, d, rd, reverse=False):
        '''
        initialization method for Simple RNN

        model - pre-instantiated dynet Model object
        d - dimensionality of input
        rd - dimensionality of hidden state
        reverse - reverse the direction of the RNN
        '''
        self.d = d
        self.rd = rd
        self.reverse = reverse

        #input to hidden
        self.W_x = model.add_parameters((rd, d))
        #hidden to hidden
        self.W_h = model.add_parameters((rd, rd))
        #bias
        self.b = model.add_parameters((rd,))
        #hidden state initialization
        self.h_0 = model.add_parameters((rd,))        
        
        self._params = [self.W_x, self.W_h, self.b, self.h_0]

    @property
    def params(self):
        return self._params

    def get_output(self, input, h_start=None):
        '''
        apply the RNN to a matrix or list of vectors
        
        input - a list of vectors of dimension d
        h_start - optional start state for continuation (default is to start at the beginning with h_0)
        '''
        W_x = parameter(self.W_x)
        W_h = parameter(self.W_h)
        b = parameter(self.b)

        #option for continuing the RNN
        if h_start is None:
            h_0 = parameter(self.h_0)
        else:
            h_0 = h_start

        #YOUR IMPLEMENTATION HERE
        # raise NotImplementedError
        if self.reverse:
            input = reversed(input)

        states = []
        for i in input:
            h_0 = nonlinearities['tanh'](W_x * i + W_h * h_0 + b)
            states.append(h_0)

        return [states]

class LSTM:
    '''
    Long Short-term Memory Network for sequence modeling
    '''
    def __init__(self, model, d, rd, reverse=False):
        '''
        initialization method for LSTM class
        
        model - pre-instantiated Dynet model object
        d - dimension of input
        rd - dimension of hidden state
        reverse - run LSTM in backward direction
        '''
        
        self.d = d
        self.rd = rd
        self.reverse = reverse

        #for all parameters, W is input to hidden and U is hidden to hidden
        
        #forget gate parameters
        self.W_f = model.add_parameters((rd, d))
        self.U_f = model.add_parameters((rd, rd))
        self.b_f = model.add_parameters((rd,))

        #input gate parameters
        self.W_i = model.add_parameters((rd, d))
        self.U_i = model.add_parameters((rd, rd))
        self.b_i = model.add_parameters((rd,))

        #output gate parameters
        self.W_o = model.add_parameters((rd, d))
        self.U_o = model.add_parameters((rd, rd))
        self.b_o = model.add_parameters((rd,))

        #hidden state parameters
        self.W_c = model.add_parameters((rd, d))
        self.U_c = model.add_parameters((rd, rd))
        self.b_c = model.add_parameters((rd,))

        #initial hidden state
        self.h_0 = model.add_parameters((rd,))
        #initial cell
        self.c_0 = model.add_parameters((rd,))

        self._params = [self.W_f, self.U_f, self.b_f,
                       self.W_i, self.U_i, self.b_i,
                       self.W_o, self.U_o, self.b_o,
                       self.W_c, self.U_c, self.b_c,
                       self.h_0, self.c_0]

    @property
    def params(self):
        return self._params
    
    def get_output(self, input, h_start=None, c_start=None):
        '''
        apply the LSTM to a matrix or list of vectors
        
        input - a list of vectors of dimension d
        h_start - optional start state for continuation (default is to start at the beginning with h_0)
        c_start - optional start cell for continuation (default is to start at the beginning with c_0)
        '''

        W_f = parameter(self.W_f)
        U_f = parameter(self.U_f)
        b_f = parameter(self.b_f)
        
        W_i = parameter(self.W_i)
        U_i = parameter(self.U_i)
        b_i = parameter(self.b_i)
        
        W_o = parameter(self.W_o)
        U_o = parameter(self.U_o)
        b_o = parameter(self.b_o)
        
        W_c = parameter(self.W_c)
        U_c = parameter(self.U_c)
        b_c = parameter(self.b_c)

        if h_start is None:
            h_0 = parameter(self.h_0)
        else:
            h_0 = h_start
        if c_start is None:
            c_0 = parameter(self.c_0)
        else:
            c_0 = c_start
        
        #IMPLEMENT YOUR LSTM CODE HERE
        # raise NotImplementedError
        if self.reverse:
            input = reversed(input)

        states = []
        cells = []
        for i in input:
            ft = nonlinearities['sigmoid'](W_f * i + U_f * h_0 + b_f)
            it = nonlinearities['sigmoid'](W_i * i + U_i * h_0 + b_i)
            ot = nonlinearities['sigmoid'](W_o * i + U_o * h_0 + b_o)
            c_0 = cmult(ft, c_0) + cmult(it, nonlinearities['tanh'](W_c * i + U_c * h_0 + b_c))
            h_0 = cmult(ot, nonlinearities['tanh'](c_0))
            states.append(h_0)
            cells.append(c_0)

        return [states, cells]
