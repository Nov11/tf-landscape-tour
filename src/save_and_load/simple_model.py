import tensorflow._api.v2.compat.v1 as tf


def make_model():
    a = tf.Variable(initial_value=[2], name='A')
    b = tf.Variable(initial_value=[4], name='B')
    pa = tf.placeholder(dtype=tf.int32, shape=[1], name='pa')
    pb = tf.placeholder(dtype=tf.int32, shape=[1], name='pb')
    ma = tf.multiply(a, pa)
    mb = tf.multiply(b, pb)
    add_op = tf.add(ma, mb, name='Add')
    return pa, pb, add_op
