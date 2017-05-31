import tensorflow as tf
import numpy as np

#fInput = open("pima-indians-diabetes.csv", "r")

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None,1])

W1 = tf.Variable(tf.random_normal([8,6]), name='weight1')
b1 = tf.Variable(tf.random_normal([6], name='bias1'))
L1 = tf.nn.sigmoid(tf.matmul(X,W1) + b1)

W2 = tf.Variable(tf.random_normal([6,4]), name='weight2')
b2 = tf.Variable(tf.random_normal([4], name='bias2'))
L2 = tf.nn.sigmoid(tf.matmul(L1,W2) + b2)

W3 = tf.Variable(tf.random_normal([4,1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1], name='bias3'))
L3 = tf.matmul(L2,W3) + b3

hypothesis = L3

cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y)
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

batch = []
for line in open("pima-indians-diabetes.csv", "r"):
    if not line: break
    splits = line.split(',')
    batch.append(line.split(','))
            
batch = np.asarray(batch)
#batch = np.asarray(line.split(','))
x_train = batch[:,:8].astype(np.float32)
x_train = x_train / x_train.max(axis = 0)
y_train = batch[:,8].reshape(768,1).astype(np.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch_index in range(1000):

        loss, hyp, _ = sess.run([cost, hypothesis, optimizer], feed_dict = {X:x_train, Y:y_train})
        weight1 = sess.run(W1)
        bias1 = sess.run(b1)
        weight2 = sess.run(W2)
        bias2 = sess.run(b2)
        weight3 = sess.run(W3)
        bias3 = sess.run(b3)
        
        pred = sess.run(tf.cast( tf.nn.sigmoid(hyp) > 0.5, dtype = tf.float32))
        accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(pred, y_train), tf.float32)))
        print batch_index, sess.run(tf.reduce_mean(loss)), accuracy
