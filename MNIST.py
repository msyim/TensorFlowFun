import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.005
training_epochs = 100
batch_size = 100
learning_rate = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,50, 0.91, staircase=True)

# Network 1

# First layer
N1W1 = tf.Variable(tf.random_normal([784, 260]))
N1b1 = tf.Variable(tf.random_normal([260]))
N1L1 = tf.nn.relu(tf.matmul(X, N1W1) + N1b1)
#N1L1BN = tf.nn.relu(tf.contrib.layers.batch_norm(N1L1, center=True, scale=True, is_training=phase))

# Second layer
N1W2 = tf.Variable(tf.random_normal([260, 260]))
N1b2 = tf.Variable(tf.random_normal([260]))
N1L2 = tf.nn.relu(tf.nn.relu(tf.matmul(N1L1, N1W2) + N1b2))
#N1L2BN = tf.nn.relu(tf.contrib.layers.batch_norm(N1L2, center=True, scale=True, is_training=phase))

# Third layer
N1W3 = tf.Variable(tf.random_normal([260, 10]))
N1b3 = tf.Variable(tf.random_normal([10]))
N1L3 = tf.matmul(N1L2, N1W3) + N1b3


# Network 2

# First layer
N2W1 = tf.Variable(tf.random_normal([784, 256]))
N2b1 = tf.Variable(tf.random_normal([256]))
N2L1 = tf.nn.relu(tf.matmul(X, N2W1) + N2b1)

# Second layer
N2W2 = tf.Variable(tf.random_normal([256, 256]))
N2b2 = tf.Variable(tf.random_normal([256]))
N2L2 = tf.nn.relu(tf.matmul(N2L1, N2W2) + N2b2)

# Third layer
N2W3 = tf.Variable(tf.random_normal([256, 10]))
N2b3 = tf.Variable(tf.random_normal([10]))
N2L3 = tf.matmul(N2L2, N2W3) + N2b3

# Network 3

# First layer
N3W1 = tf.Variable(tf.random_normal([784, 256]))
N3b1 = tf.Variable(tf.random_normal([256]))
N3L1 = tf.nn.relu(tf.matmul(X, N3W1) + N3b1)

N3W2 = tf.Variable(tf.random_normal([256, 256]))
N3b2 = tf.Variable(tf.random_normal([256]))
N3L2 = tf.nn.relu(tf.matmul(N3L1, N3W2) + N3b2)

N3W3 = tf.Variable(tf.random_normal([256, 10]))
N3b3 = tf.Variable(tf.random_normal([10]))
N3L3 = tf.matmul(N3L2, N3W3) + N3b3

# Network 4

# First layer
N4W1 = tf.Variable(tf.random_normal([784, 270]))
N4b1 = tf.Variable(tf.random_normal([270]))
N4L1 = tf.nn.relu(tf.matmul(X, N4W1) + N4b1)

N4W2 = tf.Variable(tf.random_normal([270, 270]))
N4b2 = tf.Variable(tf.random_normal([270]))
N4L2 = tf.nn.relu(tf.matmul(N4L1, N4W2) + N4b2)

N4W3 = tf.Variable(tf.random_normal([270, 10]))
N4b3 = tf.Variable(tf.random_normal([10]))
N4L3 = tf.matmul(N4L2, N4W3) + N4b3

# Network 5

# First layer
N5W1 = tf.Variable(tf.random_normal([784, 256]))
N5b1 = tf.Variable(tf.random_normal([256]))
N5L1 = tf.nn.relu(tf.matmul(X, N5W1) + N5b1)

N5W2 = tf.Variable(tf.random_normal([256, 256]))
N5b2 = tf.Variable(tf.random_normal([256]))
N5L2 = tf.nn.relu(tf.matmul(N5L1, N5W2) + N5b2)

N5W3 = tf.Variable(tf.random_normal([256, 10]))
N5b3 = tf.Variable(tf.random_normal([10]))
N5L3 = tf.matmul(N5L2, N5W3) + N5b3

# Network 6

# First layer
N6W1 = tf.Variable(tf.random_normal([784, 256]))
N6b1 = tf.Variable(tf.random_normal([256]))
N6L1 = tf.nn.relu(tf.matmul(X, N6W1) + N6b1)

N6W2 = tf.Variable(tf.random_normal([256, 256]))
N6b2 = tf.Variable(tf.random_normal([256]))
N6L2 = tf.nn.relu(tf.matmul(N6L1, N6W2) + N6b2)

N6W3 = tf.Variable(tf.random_normal([256, 10]))
N6b3 = tf.Variable(tf.random_normal([10]))
N6L3 = tf.matmul(N6L2, N6W3) + N6b3

# Network 7

# First layer
N7W1 = tf.Variable(tf.random_normal([784, 256]))
N7b1 = tf.Variable(tf.random_normal([256]))
N7L1 = tf.nn.relu(tf.matmul(X, N7W1) + N7b1)

N7W2 = tf.Variable(tf.random_normal([256, 256]))
N7b2 = tf.Variable(tf.random_normal([256]))
N7L2 = tf.nn.relu(tf.matmul(N7L1, N7W2) + N7b2)

N7W3 = tf.Variable(tf.random_normal([256, 10]))
N7b3 = tf.Variable(tf.random_normal([10]))
N7L3 = tf.matmul(N7L2, N7W3) + N7b3

# Network 8

# First layer
N8W1 = tf.Variable(tf.random_normal([784, 256]))
N8b1 = tf.Variable(tf.random_normal([256]))
N8L1 = tf.nn.relu(tf.matmul(X, N8W1) + N8b1)

N8W2 = tf.Variable(tf.random_normal([256, 256]))
N8b2 = tf.Variable(tf.random_normal([256]))
N8L2 = tf.nn.relu(tf.matmul(N8L1, N8W2) + N8b2)

N8W3 = tf.Variable(tf.random_normal([256, 10]))
N8b3 = tf.Variable(tf.random_normal([10]))
N8L3 = tf.matmul(N8L2, N8W3) + N8b3

# Network 9

# First layer
N9W1 = tf.Variable(tf.random_normal([784, 256]))
N9b1 = tf.Variable(tf.random_normal([256]))
N9L1 = tf.nn.relu(tf.matmul(X, N9W1) + N9b1)

N9W2 = tf.Variable(tf.random_normal([256, 256]))
N9b2 = tf.Variable(tf.random_normal([256]))
N9L2 = tf.nn.relu(tf.matmul(N9L1, N9W2) + N9b2)

N9W3 = tf.Variable(tf.random_normal([256, 10]))
N9b3 = tf.Variable(tf.random_normal([10]))
N9L3 = tf.matmul(N9L2, N9W3) + N9b3

# Final layer : Ensemble 
FL  = N1L3 + N2L3 + N3L3 + N4L3 + N5L3 + N6L3 + N7L3 + N8L3

# Training with the sum of the final layers is not really working (acc ~ 0.92)
# Let's train the networks separately and add them afterwards to get the prediction.

# With the second method, I can only get upto (acc ~ 0.95)

# Cost function
c1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N1L3))
c2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N2L3))
c3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N3L3))
c4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N4L3))
c5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N5L3))
c6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N6L3))
c7 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N7L3))
c8 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N8L3))
c9 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N9L3))

# Optimizer
op1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c1)
op2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c2)
op3 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c3)
op4 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c4)
op5 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c5)
op6 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c6)
op7 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c7)
op8 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c8)
op9 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c9)

# Test model and check accuracy

# Voting method.

correct_pred = tf.equal(tf.argmax(FL,1), tf.argmax(Y,1))
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
        
		total_batch = int(mnist.train.num_examples / batch_size)

		lr = 0.005
		if epoch >= 80 : lr = 0.0025
		if epoch >= 90 : lr = 0.001
		
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			l1,l2,l3,l4,l5,l6,l7,l8,_,_,_,_,_, _,_,_ = sess.run([c1,c2,c3,c4,c5,c6,c7,c8, op1,op2,op3,op4,op5,op6,op7,op8], feed_dict={X:batch_xs, Y:batch_ys, learning_rate : lr})
			avg_cost1 += (l1)/(total_batch)
			avg_cost2 += (l2)/(total_batch)
			avg_cost3 += (l3)/(total_batch)
			avg_cost4 += (l4)/(total_batch)            
			avg_cost5 += (l5)/(total_batch) 
			avg_cost6 += (l6)/(total_batch)
			avg_cost7 += (l7)/(total_batch)
			avg_cost8 += (l8)/(total_batch)
			#avg_cost9 += (l9)/(total_batch)            
			#for el in range( n1l1.shape[0]):
			#	print n1l1[0]
			#	print n1l2[0]
			#	print n1l3[0]
			#	print n1l4[0]
		
		print ('Epoch: ', '%04d' % (epoch + 1), 'c1 = ', '{:.9f}'.format(avg_cost1), 'c2 = ', '{:.9f}'.format(avg_cost2), 'c3 = ', '{:.9f}'.format(avg_cost3), 'c4 = ', '{:.9f}'.format(avg_cost4), 'c5 = ', '{:.9f}'.format(avg_cost5), 'c6 = ', '{:.9f}'.format(avg_cost6), 'c7 = ', '{:.9f}'.format(avg_cost7), 'c8 = ', '{:.9f}'.format(avg_cost8), 'c9 = ', '{:.9f}'.format(avg_cost9))
		print('Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
	print('Learning Finished!')
