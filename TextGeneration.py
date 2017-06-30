import tensorflow as tf

# Meta
INPUT='bible.txt'

# Helper fnc : returns the dictionary and the size of it given a text corpus
def makeDictionary(inputFileName):

	my_dict = {}
	rev_dict = {}
	index = 0
	for line in open(inputFileName):
		splits = line.split()
		for el in splits :
			if el not in my_dict : 
				my_dict[el] = index
				rev_dict[index] = el
				index += 1

	return index, my_dict, rev_dict

# Weight initializer helper
def weightInitializer(name, shape):
	tf.get_variable(name, shape, initializer=tf.contrib.keras.initializers.he_normal())	

# make dictionary
dict_size, my_dict, rev_dict = makeDictionary(INPUT)

# TF Hyperparameters
timesteps = 256
inputDim = dict_size
nLSTMHidden = 128

X = tf.placeholder(tf.float32, shape=[None,timesteps,inputDim])
Y = tf.placeholder(tf.float32, shape=[None,inputDim])

embedW = weightInitializer("WordEmbeddingW", [inputDim, 256])
embedB = weightInitializer("WordEmbeddingB", [256])
LSTMOutW = weightInitializer("LSTMOutW", [nLSTMHidden, inputDim])
LSTMOutB = weightInitializer("LSTMOutB", [inputDim])

def LSTMmodel(X):
	# current shape of X : [batch, timesteps, inputDim]
	# but we want 'timesteps' tensors of shape : [batch, inputDim]
	X = tf.unstack(X, timesteps, 1)

	LSTM_cell = tf.contrib.rnn.BasicLSTMCell(nLSTMHidden)

	# predict
	outputs, states = tf.contrib.rnn.static_rnn(LSTM_cell, X, dtype=tf.float32)

	# there are n_input outputs but
	# we only want the last output
	return tf.matmul(outputs[-1], LSTMOutW) + LSTMOutB

pred = LSTMmodel(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)


