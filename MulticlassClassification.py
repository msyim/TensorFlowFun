import tensorflow as tf
import numpy as np

# x positive = 1
# x negtiave = 0

x_train = [[1,-11], [5,3], [-4,2], 
						[-11,3], [-1,-75], [2,-23], 
						[15,2],[-13,-24]]
y_train = [[0,0,0,1],[1,0,0,0],[0,1,0,0],
						[0,1,0,0],[0,0,1,0],[0,0,0,1],
						[1,0,0,0],[0,0,1,0]]

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,4])

W = tf.Variable(tf.random_normal([2,4]), name='weight')
b = tf.Variable(tf.random_normal([4]), name='bias')

# Method 1
hypothesis = tf.matmul(X,W) + b
cost = tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y)
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# Cannot put these in the graph
#predicted = np.argmax(hypothesis, axis=1)
#accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, np.argmax(Y, axis=1)), dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(20000):
		loss, _, hypo = sess.run([cost,optimizer,hypothesis], feed_dict={X:x_train, Y:y_train})
		predicted = sess.run(tf.argmax(hypo, axis=1))
		accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y_train, axis=1)), dtype=tf.float32)))
		print step, loss, hypo, predicted, accuracy
		
