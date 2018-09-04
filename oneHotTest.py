import tensorflow as tf
from random import randint
import numpy as np

dims = 8
pos1  = 0
one_hot = np.array([0]*(dims), dtype = np.float32)
one_hot[0] = 0

logits = tf.random_uniform([dims], maxval=3, dtype=tf.float32)

res1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = tf.constant(pos1))
res2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels= one_hot)

with tf.Session() as sess:
    a, b = sess.run([res1, res2])
    print("Sparse label result: {}".format(a))
    print("One hot label result: {}".format(b))
    print(a == b)