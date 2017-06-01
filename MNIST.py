import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
training_epochs = 200
batch_size = 100

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

# Network 1

# First layer
N1W1 = tf.Variable(tf.random_normal([784, 500]))
N1b1 = tf.Variable(tf.random_normal([500]))
N1L1 = tf.nn.relu(tf.matmul(X, N1W1) + N1b1)

# Second layer
N1W2 = tf.Variable(tf.random_normal([500, 300]))
N1b2 = tf.Variable(tf.random_normal([300]))
N1L2 = tf.nn.relu(tf.matmul(N1L1, N1W2) + N1b2)

# Third layer
N1W3 = tf.Variable(tf.random_normal([300, 100]))
N1b3 = tf.Variable(tf.random_normal([100]))
N1L3 = tf.nn.relu(tf.matmul(N1L2, N1W3) + N1b3)

# Fourth layer
N1W4 = tf.Variable(tf.random_normal([100, 10]))
N1b4 = tf.Variable(tf.random_normal([10]))
N1L4 = tf.matmul(N1L3, N1W4) + N1b4

# Network 2

# First layer
N2W1 = tf.Variable(tf.random_normal([784, 300]))
N2b1 = tf.Variable(tf.random_normal([300]))
N2L1 = tf.nn.relu(tf.matmul(X, N2W1) + N2b1)

# Second layer
N2W2 = tf.Variable(tf.random_normal([300, 150]))
N2b2 = tf.Variable(tf.random_normal([150]))
N2L2 = tf.nn.relu(tf.matmul(N2L1, N2W2) + N2b2)

# Third layer
N2W3 = tf.Variable(tf.random_normal([150, 10]))
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

# Final layer : Ensemble 
FL  = N1L4 + N2L3 + N3L3

# Training with the sum of the final layers is not really working (acc ~ 0.92)
# Let's train the networks separately and add them afterwards to get the prediction.

# With the second method, I can only get upto (acc ~ 0.95)

# Cost function
cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N1L4))
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N2L3))
cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=N3L3))

# Optimizer
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost1)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost2)
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost3)

# Test model and check accuracy

# Voting method.

correct_pred = tf.equal(tf.argmax(FL,1), tf.argmax(Y,1))
accuracy  = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples / batch_size)
		
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			l1, l2, l3, _, _, _= sess.run([cost1, cost2, cost3, optimizer1, optimizer2, optimizer3], feed_dict={X:batch_xs, Y:batch_ys})
			avg_cost += (l1+l2+l3)/(3*total_batch)
		
		print ('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
		print('Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
	print('Learning Finished!')

