import tensorflow as tf

# X and Y data
#x_train = [1, 2, 3]
#y_train = [1, 2, 3]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
#    sess.run(train)

    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={x_train: [1,2,3], y_train:[1,2,3]})
    if step == 0:
      print("step,  cost,   W,  b")
    if step % 20 == 0:
      print(step, cost_val, W_val, b_val)

