#!/usr/bin/env python
# coding=utf-8
# Import packages
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import load_data
import matplotlib.pyplot as plt
import argparse
#matplotlib inline 
print ("Packages imported")

def parse_args1():
    parser = argparse.ArgumentParser()
    parser.add_argument("pr")
    #parser.add_argument("-name","--names_print")
    #parser.add_argument("-inc","--increments")
    #print parser.pr
    return parser.parse_args()
args1 = parse_args1()
# Load MNIST, our beloved friend
mnist =  load_data.read_data_sets("/mnt/d/workspace/ubuntu/workspace/ocr/"+args1.pr,
                   "/mnt/d/workspace/ubuntu/workspace/ocr/"+args1.pr, one_hot=True,validation_size=1)
trainimgs, trainlabels, testimgs, testlabels \
 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels 
ntrain, ntest, dim, nclasses \
 = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]
print ("MNIST loaded")

# Recurrent neural network 
diminput  = 28
dimhidden = 128
dimoutput = nclasses
nsteps    = 28
weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])), 
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}
def _RNN(_X, _istate, _W, _b, _nsteps, _name):
    # 1. Permute input from [batchsize, nsteps, diminput] => [nsteps, batchsize, diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # 2. Reshape input to [nsteps*batchsize, diminput] 
    _X = tf.reshape(_X, [-1, diminput])
    # 3. Input layer => Hidden layer
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 4. Splite data to 'nsteps' chunks. An i-th chunck indicates i-th batch data 
    _Hsplit = tf.split(0, _nsteps, _H) 
    # 5. Get LSTM's final output (_O) and state (_S)
    #    Both _O and _S consist of 'batchsize' elements
    with tf.variable_scope(_name):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_O, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, initial_state=_istate)
    # 6. Output
    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']    
    # Return! 
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O 
    }
print ("Network ready")

learning_rate = 0.001
x      = tf.placeholder("float", [None, nsteps, diminput])
istate = tf.placeholder("float", [None, 2*dimhidden]) #state & cell => 2x n_hidden
y      = tf.placeholder("float", [None, dimoutput])
myrnn  = _RNN(x, istate, weights, biases, nsteps, 'basic')
pred   = myrnn['O']
cost   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) 
optm   = tf.train.AdamOptimizer(learning_rate).minimize(cost) # Adam Optimizer
accr   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))
init   = tf.initialize_all_variables()
print ("Network Ready!")


batch_size      = 1
display_step    = 1
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('./logs/', graph=sess.graph)
# added by siler :to save model
saver = tf.train.Saver()
print ("Start optimization")
continue_test = raw_input("continue test--True or False: ")
saver.restore(sess,'./logs/mnist.tfmodel-99')
while continue_test == '1':
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))

    prediction,y_= sess.run([tf.argmax(pred,1),tf.argmax(y,1)],feed_dict=
              {x:batch_xs,y:batch_ys,istate: np.zeros((batch_size, 2*dimhidden))})
    print " Prediction is: ",prediction
    print "Y           is:  ",y_
    print "shape of pre:   ",prediction.shape
    continue_test = raw_input("continue test:True or False ")

print ("Optimization Finished.")

