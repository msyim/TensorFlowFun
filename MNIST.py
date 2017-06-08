import tensorflow as tf
import random
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#learning_rate = 0.005
training_epochs = 150
batch_size = 100
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,50, 0.91, staircase=True)

# Network 1

# First layer
N1W1 = tf.get_variable("N1W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
N1b1 = tf.Variable(tf.random_normal([512]))
N1L1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, N1W1) + N1b1), keep_prob=keep_prob)

# Second layer
N1W2 = tf.get_variable("N1W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N1b2 = tf.Variable(tf.random_normal([512]))
N1L2 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N1L1, N1W2) + N1b2)), keep_prob=keep_prob)

# Third layer
N1W3 = tf.get_variable("N1W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N1b3 = tf.Variable(tf.random_normal([512]))
N1L3 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N1L2, N1W3) + N1b3)), keep_prob=keep_prob)

# Fourth layer
N1W4 = tf.get_variable("N1W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N1b4 = tf.Variable(tf.random_normal([512]))
N1L4 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N1L3, N1W4) + N1b4)), keep_prob=keep_prob)

# Fifth layer
N1W5 = tf.get_variable("N1W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
N1b5 = tf.Variable(tf.random_normal([10]))
N1L5 = tf.matmul(N1L4, N1W5) + N1b5



# Network 2

# First layer
N2W1 = tf.get_variable("N2W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
N2b1 = tf.Variable(tf.random_normal([512]))
N2L1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, N2W1) + N2b1), keep_prob=keep_prob)

# Second layer
N2W2 = tf.get_variable("N2W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N2b2 = tf.Variable(tf.random_normal([512]))
N2L2 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N2L1, N2W2) + N2b2)), keep_prob=keep_prob)

# Third layer
N2W3 = tf.get_variable("N2W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N2b3 = tf.Variable(tf.random_normal([512]))
N2L3 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N2L2, N2W3) + N2b3)), keep_prob=keep_prob)

# Fourth layer
N2W4 = tf.get_variable("N2W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N2b4 = tf.Variable(tf.random_normal([512]))
N2L4 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N2L3, N2W4) + N2b4)), keep_prob=keep_prob)

# Fifth layer
N2W5 = tf.get_variable("N2W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
N2b5 = tf.Variable(tf.random_normal([10]))
N2L5 = tf.matmul(N2L4, N2W5) + N2b5

# Network 3

# First layer
N3W1 = tf.get_variable("N3W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
N3b1 = tf.Variable(tf.random_normal([512]))
N3L1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, N3W1) + N3b1), keep_prob=keep_prob)

# Second layer
N3W2 = tf.get_variable("N3W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N3b2 = tf.Variable(tf.random_normal([512]))
N3L2 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N3L1, N3W2) + N3b2)), keep_prob=keep_prob)

# Third layer
N3W3 = tf.get_variable("N3W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N3b3 = tf.Variable(tf.random_normal([512]))
N3L3 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N3L2, N3W3) + N3b3)), keep_prob=keep_prob)

# Fourth layer
N3W4 = tf.get_variable("N3W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N3b4 = tf.Variable(tf.random_normal([512]))
N3L4 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N3L3, N3W4) + N3b4)), keep_prob=keep_prob)

# Fifth layer
N3W5 = tf.get_variable("N3W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
N3b5 = tf.Variable(tf.random_normal([10]))
N3L5 = tf.matmul(N3L4, N3W5) + N3b5


# Network 4

# First layer
N4W1 = tf.get_variable("N4W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
N4b1 = tf.Variable(tf.random_normal([512]))
N4L1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, N4W1) + N4b1), keep_prob=keep_prob)

# Second layer
N4W2 = tf.get_variable("N4W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N4b2 = tf.Variable(tf.random_normal([512]))
N4L2 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N4L1, N4W2) + N4b2)), keep_prob=keep_prob)

# Third layer
N4W3 = tf.get_variable("N4W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N4b3 = tf.Variable(tf.random_normal([512]))
N4L3 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N4L2, N4W3) + N4b3)), keep_prob=keep_prob)

# Fourth layer
N4W4 = tf.get_variable("N4W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
N4b4 = tf.Variable(tf.random_normal([512]))
N4L4 = tf.nn.dropout(tf.nn.relu(tf.nn.relu(tf.matmul(N4L3, N4W4) + N4b4)), keep_prob=keep_prob)

# Fifth layer
N4W5 = tf.get_variable("N4W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
N4b5 = tf.Variable(tf.random_normal([10]))
N4L5 = tf.matmul(N4L4, N4W5) + N4b5

# Final layer : Ensemble 
# FL  = N1L3 + N2L3 + N3L3 + N4L3 + N5L3 + N6L3 + N7L3 + N8L3 + N9L3 + N0L2
FL = N1L5 + N2L5 + N3L5 + N4L5

# Training with the sum of the final layers is not really working (acc ~ 0.92)
# Let's train the networks separately and add them afterwards to get the prediction.

# With the second method, I can only get upto (acc ~ 0.95)

# Cost function
c1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N1L5))
c2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N2L5))
c3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N3L5))
c4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N4L5))
#c5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N5L3))
#c6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N6L3))
#c7 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N7L3))
#c8 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N8L3))
#c9 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N9L3))
#c0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N0L2))

# Optimizer
op1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c1)
op2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c2)
op3 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c3)
op4 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c4)
#op5 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c5)
#op6 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c6)
#op7 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c7)
#op8 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c8)
#op9 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c9)
#op0 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(c0)

# Test model and check accuracy

# Voting method.
prediction = tf.argmax(FL,1)
correct_pred = tf.equal(prediction, tf.argmax(Y,1))
accuracy  = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost1 = 0
		avg_cost2 = 0
		avg_cost3 = 0
		avg_cost4 = 0
		avg_cost5 = 0
		avg_cost6 = 0
		avg_cost7 = 0
		avg_cost8 = 0
		avg_cost9 = 0
		avg_cost0 = 0
        
		total_batch = int(mnist.train.num_examples / batch_size)

		lr = 0.004 * ( 1 - (float(epoch)/float(training_epochs)))
		#lr = 0.001
		print("learning rate: %f" % lr)

		stat_grid = []
		for _ in range(10):
			sub_list = [0,0,0,0,0,0,0,0,0,0]
			stat_grid.append(sub_list)
        	
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			#l1,l2,l3,l4,l5,l6,l7,l8,l9,l0,_,_,_,_,_,_, _,_,_,_, pred = sess.run([c1,c2,c3,c4,c5,c6,c7,c8,c9,c0, op1,op2,op3,op4,op5,op6,op7,op8,op9,op0, prediction], feed_dict={X:batch_xs, Y:batch_ys, learning_rate : lr, keep_prob : 0.96})
			l1,l2,l3,l4,_,_,_,_,pred = sess.run([c1,c2,c3,c4,op1,op2,op3,op4,prediction], feed_dict={X:batch_xs, Y:batch_ys, learning_rate : lr, keep_prob : 0.95})
			avg_cost1 += (l1)/(total_batch)
			avg_cost2 += (l2)/(total_batch)
			avg_cost3 += (l3)/(total_batch)
			avg_cost4 += (l4)/(total_batch)            
			#avg_cost5 += (l5)/(total_batch) 
			#avg_cost6 += (l6)/(total_batch)
			#avg_cost7 += (l7)/(total_batch)
			#avg_cost8 += (l8)/(total_batch)
			#avg_cost9 += (l9)/(total_batch)            
			#avg_cost0 += (l0)/(total_batch)            
			#for el in range( n1l1.shape[0]):
			#	print n1l1[0]
			#	print n1l2[0]
			#	print n1l3[0]
			#	print n1l4[0]

			g_truth = np.argmax(batch_ys, axis=1)
			for index in range(len(pred)):
				if pred[index] != g_truth[index]:
					stat_grid[int(pred[index])][int(g_truth[index])] = stat_grid[int(pred[index])][int(g_truth[index])] + 1
		
		print ('Epoch: ', '%04d' % (epoch + 1), 'c1 = ', '{:.9f}'.format(avg_cost1), 'c2 = ', '{:.9f}'.format(avg_cost2), 'c3 = ', '{:.9f}'.format(avg_cost3), 'c4 = ', '{:.9f}'.format(avg_cost4), 'c5 = ', '{:.9f}'.format(avg_cost5), 'c6 = ', '{:.9f}'.format(avg_cost6), 'c7 = ', '{:.9f}'.format(avg_cost7), 'c8 = ', '{:.9f}'.format(avg_cost8), 'c9 = ', '{:.9f}'.format(avg_cost9), 'c10 = ', '{:.9f}'.format(avg_cost0))
		print('Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))

		for index in range(10):
			outputString = ""
			for index2 in range(10):
				outputString += (str(stat_grid[index][index2]) + '\t')
			print outputString
            
	print('Learning Finished!')
