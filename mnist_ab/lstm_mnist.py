#!/usr/bin/env python
# coding=utf-8
# Import packages
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import load_data
#import matplotlib.pyplot as plt
#matplotlib inline 
print ("Packages imported")

# Load MNIST, our beloved friend
mnist =  load_data.read_data_sets("/home/a/workspace/ssd/mnist_ab/","/home/a/workspace/ssd/mnist_ab_test", one_hot=False)
trainimgs, trainlabels, testimgs, testlabels \
 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels 
ntrain, ntest, dim, nclasses \
 = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]
print "ntrain:  ",ntrain
print "dim:     ",dim
print "nclasses: ",nclasses
print ("MNIST loaded")

# Recurrent neural network 
diminput  = 28
dimhidden = 128
dimoutput = nclasses
nsteps    = 28
num_layers = 2
weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])), 
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}
def _RNN(_X, batch_size, _W, _b,num_layers,_nsteps, _name):
#def _RNN(_X, _istate, _W, _b,_nsteps, _name):
    # 1. Permute input from [batchsize, nsteps, diminput] => [nsteps, batchsize, diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # 2. Reshape input to [nsteps*batchsize, diminput] 
    _X = tf.reshape(_X, [-1, diminput])
    # 3. Input layer => Hidden layer
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 4. Splite data to 'nsteps' chunks. An i-th chunck indicates i-th batch data 
    _Hsplit = tf.split(0, _nsteps, _H) 
    #_Hsplit shape is (_nsteps,batch_size,dimhidden)

    # 5. Get LSTM's final output (_O) and state (_S)
    #    Both _O and _S consist of 'batchsize' elements
    with tf.variable_scope(_name):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden,\
                                     forget_bias=1.0)
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers,\
                                     state_is_tuple=True)
        print "print tf.nn.rnn......"
        state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
        _LSTM_O, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, \
                                     initial_state=state)
        #_LSTM_O, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, initial_state=_istate)
        print "all is done..........."
    # 6. Output
    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']    
    # Return! 
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O 
    }
print ("Network ready")

batch_size      = 1000
learning_rate = 0.01
#x      = tf.placeholder("float", [None, nsteps, diminput])
x      = tf.placeholder("float", [batch_size, nsteps, diminput])
#istate = tf.placeholder("float", [None, 2*dimhidden]) #state & cell => 2x n_hidden
y      = tf.placeholder("float", [None, dimoutput])
#myrnn  = _RNN(x, istate, weights, biases,num_layers,nsteps, 'basic')
myrnn  = _RNN(x,batch_size,weights,biases,num_layers,nsteps,'Basic')
pred   = myrnn['O']
cost   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) 
optm   = tf.train.AdamOptimizer(learning_rate).minimize(cost) # Adam Optimizer
accr   = tf.reduce_mean(tf.cast(tf.equal(pred,y), tf.float32))
init   = tf.initialize_all_variables()
print ("All is  Ready!")


training_epochs = 900

display_step    = 20
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('./logs/', graph=sess.graph)
print ("Start optimization")
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
        # Fit training using batch data
        #sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*dimhidden))})
        feed_dict = {x:batch_xs,y:batch_ys}
        sess.run(optm, feed_dict=feed_dict)
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict=feed_dict)/total_batch
        
        #print "COST_pred shape is :",pred.shape
    # Display logs per epoch step
    if epoch % display_step == 0: 
        print "*******************************************************************************"
        print "optm_y shape is :",batch_ys.shape
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feed_dict = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feed_dict)
        print (" Training accuracy: %.3f" % (train_acc))
        batch_test_xs,batch_test_ys = mnist.test.next_batch(batch_size)
        batch_test_xs  = batch_test_xs.reshape((batch_size,nsteps,diminput))
        #testimgs = testimgs.reshape((batch_size, nsteps, diminput))
        feed_dict={x: batch_test_xs, y:batch_test_ys}
        test_acc = sess.run(accr, feed_dict=feed_dict)
        print (" Test accuracy: %.3f" % (test_acc))
print ("Optimization Finished.")