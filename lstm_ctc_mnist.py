#!/usr/bin/env python
# coding=utf-8
# Import packages
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
try:
    import tensorflow.contrib.ctc as ctc
except ImportError:
    from tensorflow import nn as ctc
import numpy as np
import load_data

print ("Packages imported")

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where 
                         each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), xrange(len(seq))))
        #print "length is :   ",seq[0].shape[0],"  seq is:   ",seq[0][0]," seq type is: ",type(seq[0][0])
        values.extend(seq)
        #print "seq is :                               ",seq
        #values.extend([seq[0][1]])

    indicbbbes = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def _RNN(_X,batch_size, _W, _b,num_layers,_nsteps, _name):
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
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, \
                                         forget_bias=1.0)
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers,\
                                         state_is_tuple=True)
        state     = lstm_cell.zero_state(batch_size,dtype=tf.float32)
        _LSTM_O, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, \
                                         initial_state=state)
    # 6. Output
    _O = [tf.matmul(x, _W['out']) + _b['out'] for x in _LSTM_O]
    _O = tf.pack(_O)
    # Return!
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O
    }

# Load MNIST, our beloved friend
mnist = input_data.read_data_sets("data/",one_hot=False)
trainimgs, trainlabels, testimgs, testlabels = mnist.train.images,\
                                               mnist.train.labels,\
                                               mnist.test.images,\
                                               mnist.test.labels

ntrain, ntest, dim \
   = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1]
print "ntrain:  ",ntrain
print "dim:     ",dim
nclasses = 10
print "nclasses: ",nclasses

print ("MNIST loaded")

# Training params
training_epochs =  300
batch_size      =  1000
display_step    =  20
learning_rate   =  0.01
num_layers      =  2

# Recurrent neural network params
diminput = 28
dimhidden = 4
# here we add the blank label
dimoutput = nclasses+1
print "dimoutput:   ",dimoutput
nsteps = 28

graph = tf.Graph()
with graph.as_default():
    weights = {
        'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
        'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([dimhidden])),
        'out': tf.Variable(tf.random_normal([dimoutput]))
    }


    #**************************************************
    # will be used in CTC_LOSS
    #x = tf.placeholder(tf.float32, [None, nsteps, diminput])
    x = tf.placeholder(tf.float32, [batch_size, nsteps, diminput])
    istate = tf.placeholder(tf.float32, [batch_size, 2*dimhidden]) #state & cell => 2x n_hidden
    #istate = tf.placeholder(tf.float32, [None, 2*dimhidden]) #state & cell => 2x n_hidden
    #y  = tf.placeholder("float",[None,dimoutput])
    y = tf.sparse_placeholder(tf.int32)
    # 1d array of size [batch_size]
    # Seq len indicates the quantity of true data in the input, since when working with batches we have to pad with zeros to fit the input in a matrix
    seq_len = tf.placeholder(tf.int32, [None])

    myrnn = _RNN(x,batch_size, weights, biases,num_layers,nsteps, 'basic')
    pred = myrnn['O']
    #**************************************************
    # we add ctc module

    loss = ctc.ctc_loss(pred, y, seq_len)

    cost = tf.reduce_mean(loss)
    #cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))

    # Adam Optimizer
    optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #Decode the best path
    decoded, log_prob = ctc.ctc_greedy_decoder(pred, seq_len)
    accr = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))
    #accr  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
    init  = tf.initialize_all_variables()
    print ("Network Ready!")


with tf.Session(graph=graph) as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('./logs/', graph=sess.graph)
    print ("Start optimization")
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)*2
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
            # Fit training using batch data
            #feed_dict={x: batch_xs, y: sparse_tuple_from([[value] for value in batch_ys]),\
            #                             istate: np.zeros((batch_size, 2*dimhidden)), \
            #                             seq_len: [nsteps for _ in xrange(batch_size)]}
            feed_dict={x: batch_xs, y: sparse_tuple_from([[value] for value in batch_ys]),\
                                         seq_len: [nsteps for _ in xrange(batch_size)]} 
            '''
            feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*dimhidden))}
            '''
            _, batch_cost = sess.run([optm, cost], feed_dict=feed_dict)
            # Compute average loss
            avg_cost += batch_cost*batch_size
            #print "COST_pred shape is :",pred.shape
        avg_cost /= len(trainimgs)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))


            train_acc = sess.run(accr, feed_dict=feed_dict)
            print ("    Training    label    error   rate:   %.3f" % (train_acc))
            #testimgs = testimgs.reshape((ntest, nsteps, diminput))
            batch_txs,batch_tys = mnist.test.next_batch(batch_size)
            batch_txs = batch_txs.reshape((batch_size,nsteps,diminput))

            feed_dict={x:batch_txs, y: sparse_tuple_from([[value] for value in batch_tys]), \
                                 seq_len: [nsteps for _ in xrange(batch_size)]}
            test_acc = sess.run(accr, feed_dict=feed_dict)
            print (" Test label error rate: %.3f" % (test_acc))
print ("Optimization Finished.")
