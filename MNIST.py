import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.005
training_epochs = 200
batch_size = 100


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
N3W1 = tf.Variable(tf.random_normal([784, 250]))
N3b1 = tf.Variable(tf.random_normal([250]))
N3L1 = tf.nn.relu(tf.matmul(X, N3W1) + N3b1)

N3W2 = tf.Variable(tf.random_normal([250, 250]))
N3b2 = tf.Variable(tf.random_normal([250]))
N3L2 = tf.nn.relu(tf.matmul(N3L1, N3W2) + N3b2)

N3W3 = tf.Variable(tf.random_normal([250, 10]))
N3b3 = tf.Variable(tf.random_normal([10]))
N3L3 = tf.matmul(N3L2, N3W3) + N3b3

# Network 4

# First layer
N4W1 = tf.Variable(tf.random_normal([784, 256]))
N4b1 = tf.Variable(tf.random_normal([256]))
N4L1 = tf.nn.relu(tf.matmul(X, N4W1) + N4b1)

N4W2 = tf.Variable(tf.random_normal([256, 256]))
N4b2 = tf.Variable(tf.random_normal([256]))
N4L2 = tf.nn.relu(tf.matmul(N4L1, N4W2) + N4b2)

N4W3 = tf.Variable(tf.random_normal([256, 10]))
N4b3 = tf.Variable(tf.random_normal([10]))
N4L3 = tf.matmul(N4L2, N4W3) + N4b3

# Final layer : Ensemble 
FL  = N1L3 + N2L3 + N3L3 + N4L3

# Training with the sum of the final layers is not really working (acc ~ 0.92)
# Let's train the networks separately and add them afterwards to get the prediction.

# With the second method, I can only get upto (acc ~ 0.95)

# Ensemble of 4 networks can get up to acc ~ 0.974

# Cost function
cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N1L3))
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N2L3))
cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N3L3))
cost4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N4L3))

# Optimizer
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost1)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost2)
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost3)
optimizer4 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost4)

# Test model and check accuracy
correct_pred = tf.equal(tf.argmax(FL,1), tf.argmax(Y,1))
accuracy  = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost1 = 0
		avg_cost2 = 0
		avg_cost3 = 0
		avg_cost4 = 0        
		total_batch = int(mnist.train.num_examples / batch_size)
		
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			l1, l2, l3, l4, _, _, _, _ = sess.run([cost1, cost2, cost3, cost4, optimizer1, optimizer2, optimizer3, optimizer4], feed_dict={X:batch_xs, Y:batch_ys})
			avg_cost1 += (l1)/(total_batch)
			avg_cost2 += (l2)/(total_batch)
			avg_cost3 += (l3)/(total_batch)
			avg_cost4 += (l4)/(total_batch)            
		
		print ('Epoch: ', '%04d' % (epoch + 1), 'cost1 = ', '{:.9f}'.format(avg_cost1), 'cost2 = ', '{:.9f}'.format(avg_cost2), 'cost3 = ', '{:.9f}'.format(avg_cost3), 'cost4 = ', '{:.9f}'.format(avg_cost4))
		print('Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
	print('Learning Finished!')


