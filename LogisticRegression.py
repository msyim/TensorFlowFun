import tensorflow as tf

# x positive = 1
# x negtiave = 0

x_train = [[1,-11], [5,3], [-4,2], [-11,3], [-1,75], [2,-23], [5,-2]]
y_train = [[1],[1],[0],[0],[0],[1],[1]]

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Method 1
hypothesis = tf.matmul(X,W) + b
cost = tf.nn.sigmoid_cross_entropy_with_logits(hypothesis, Y)
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Method 2
hypothesis2 = tf.sigmoid(tf.matmul(X,W) + b)
cost2 = -tf.reduce_mean(Y * tf.log(hypothesis2) + (1-Y)*tf.log(1-hypothesis2))
optimizer2 = tf.train.GradientDescentOptimizer(0.001).minimize(cost2)
predicted2 = tf.cast(hypothesis2 > 0.5, dtype=tf.float32)
accuracy2 = tf.reduce_mean(tf.cast(tf.equal(predicted2, Y), dtype=tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(20000):
		loss, _, pred, acc  = sess.run([cost,optimizer,predicted,accuracy], feed_dict={X:x_train, Y:y_train})
		#loss, _, pred, acc, hypo  = sess.run([cost2,optimizer2,predicted2,accuracy2, hypothesis2], feed_dict={X:x_train, Y:y_train})
		print step, loss, pred, acc
		
