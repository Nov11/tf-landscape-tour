import tensorflow._api.v2.compat.v1 as tf
import numpy as np


def make_model():
    a = tf.Variable(initial_value=np.arange(0, 1024), name='A', shape=[1024], dtype=tf.int32)
    b = tf.Variable(initial_value=np.arange(5000, 6024), name='B', shape=[1024], dtype=tf.int32)
    pa = tf.placeholder(dtype=tf.int32, shape=[1024], name='pa')
    pb = tf.placeholder(dtype=tf.int32, shape=[1024], name='pb')
    ma = tf.multiply(a, pa)
    mb = tf.multiply(b, pb)
    add_op = tf.add(ma, mb, name='Add')
    return pa, pb, add_op
