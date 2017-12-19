import tensorflow as tf
import sys

with tf.device('/cpu:0'):
    a = 2

    b = 3

    x = tf.add(a, b)

    y = tf.multiply(a, b)

    useless = tf.multiply(a, x)

    z = tf.pow(y, x)

with tf.Session() as sess:
    z = sess.run(z)

