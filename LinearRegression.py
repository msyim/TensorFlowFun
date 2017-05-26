import tensorflow as tf

x_train = [1.,4.]
y_train = [9.,33.]

# Build a graph
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X*W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# Start a session with the graph built above.
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(20000):
		cost1, _ = sess.run([cost,optimizer], feed_dict={X:x_train, Y:y_train})
		if step % 100 == 0:
			print step, cost1, sess.run(W), sess.run(b)
