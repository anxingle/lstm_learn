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
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
    	#print "Seq is :               ",seq
        indices.extend(zip([n]*(seq[0].shape[0]), xrange((seq[0].shape[0])))
        #print "length is : ",seq[0].shape
        values.extend([seq[0][0]])
        values.extend([seq[0][0]])

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

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
    _O = [tf.matmul(x, _W['out']) + _b['out'] for x in _LSTM_O]
    _O = tf.pack(_O)
    # Return!
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O
    }

# Load MNIST, our beloved friend
#mnist = input_data.read_data_sets("../../DataSets",one_hot=False)
mnist = load_data.read_data_sets("/home/a/workspace/ssd/DataSets/mnist_2/",\
                      "/home/a/workspace/ssd/DataSets/mnist_2/",one_hot=False)
trainimgs, trainlabels, testimgs, testlabels = mnist.train.images,\
                                               mnist.train.labels,\
                                               mnist.test.images,\
                                               mnist.test.labels

ntrain, ntest, dim = trainimgs.shape[0],\
                     testimgs.shape[0],\
                     trainimgs.shape[1],\
                     #trainlabels.shape[1]
nclasses = 10
print "ntrain:    ",ntrain
print "ntest :    ",ntest
print "dim:       ",dim
print "classes:   ",trainlabels.shape

print ("MNIST loaded")

# Training params
training_epochs = 1500
batch_size = 10
display_step = 1
learning_rate = 0.01

# Recurrent neural network params
diminput = 28
dimhidden = 128
# here we add the blank label
dimoutput = nclasses + 1
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
    x = tf.placeholder(tf.float32, [None, nsteps, diminput])
    shape_x = tf.shape(x)

    istate = tf.placeholder(tf.float32, [None, 2*dimhidden]) #state & cell => 2x n_hidden
    # [batch_size , 1]
    y = tf.sparse_placeholder(tf.int32)
    # 1d array of size [batch_size]
    # Seq len indicates the quantity of true data in the input, since when working with batches we have to pad with zeros to fit the input in a matrix
    seq_len = tf.placeholder(tf.int32, [None])

    myrnn = _RNN(x, istate, weights, biases, nsteps, 'basic')
    pred = myrnn['O']
    # [28 25 11] [nsteps, batch_size,num_nclass]
    shape_Pred = tf.shape(pred)
    #**************************************************
    # we add ctc module
    
    loss = ctc.ctc_loss(pred, y, seq_len)
    shape_y = tf.shape(y)

    cost = tf.reduce_mean(loss)

    # Adam Optimizer
    optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #Decode the best path
    decoded, log_prob = ctc.ctc_greedy_decoder(pred, seq_len)


    accr = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))
    init = tf.initialize_all_variables()
    print ("Network Ready!")


with tf.Session(graph=graph) as sess:
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
            #print "shape of batch_xs is :     ",batch_xs.shape
            # Fit training using batch data
            feed_dict={x: batch_xs, y: sparse_tuple_from([[value] for value in batch_ys]), \
                        istate: np.zeros((batch_size, 2*dimhidden)), \
                        seq_len: [nsteps for _ in xrange(batch_size)]}

            _, batch_cost = sess.run([optm, cost], feed_dict=feed_dict)
            # Compute average loss
            avg_cost += batch_cost*batch_size
            #print "COST_pred shape is :",pred.shape
        avg_cost /= len(trainimgs)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))


            train_acc = sess.run(accr, feed_dict=feed_dict)
            
            #prediction = sess.run(decoded[0],feed_dict=feed_dict)
            print "  "
            print "**************************XXXXXXXXXXXXXXXXX: "
            print "Shape of x is :   ",sess.run(shape_x,feed_dict=feed_dict)
            print "Shape of y is :   ",sess.run(shape_y,feed_dict=feed_dict)
            print "Shape of pred :   ",sess.run(shape_Pred,feed_dict=feed_dict)
            #print "prediction is :   ",prediction

            print (" Training label error rate: %.3f" % (train_acc))
            testimgs = testimgs.reshape((ntest, nsteps, diminput))

            feed_dict={x: testimgs, y: sparse_tuple_from([[value] for value in testlabels]), istate: np.zeros((ntest, 2*dimhidden)), seq_len: [nsteps for _ in xrange(len(testimgs))]}
            test_acc = sess.run(accr, feed_dict=feed_dict)
            print (" Test label error rate: %.3f" % (test_acc))
            p_xs, p_ys = mnist.train.next_batch(batch_size)
            p_xs = p_xs.reshape((batch_size,nsteps,diminput))
            '''
            feed_dict ={x:p_xs,y:sparse_tuple_from([[value] for value in p_ys]), 
                       istate: np.zeros((ntest, 2*dimhidden)),
                       seq_len: [nsteps for _ in xrange(batch_size)]}
            prediction = sess.run(decoded[0],feed_dict=feed_dict)
            print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX: ",prediction
            '''
print ("Optimization Finished.")