import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import tensorflow_recommenders_addons.dynamic_embedding as de


def make_base_model():
    a = tf.Variable(initial_value=np.ones([512, 8]), name='A', dtype=tf.int32)
    b = tf.Variable(initial_value=np.ones([512, 8]), name='B', dtype=tf.int32)
    pa = tf.placeholder(dtype=tf.int32, shape=[None, 512], name='pa')
    pb = tf.placeholder(dtype=tf.int32, shape=[None, 512], name='pb')
    ma = tf.matmul(pa, a)
    mb = tf.matmul(pb, b)
    add_op = tf.add(ma, mb, name='Add')
    return pa, pb, add_op


def make_target_model():
    a = tf.Variable(initial_value=np.zeros([512, 8]), name='A', dtype=tf.int32)
    b = tf.Variable(initial_value=np.zeros([512, 8]), name='B', dtype=tf.int32)
    pa = tf.placeholder(dtype=tf.int32, shape=[None, 512], name='pa')
    pb = tf.placeholder(dtype=tf.int32, shape=[None, 512], name='pb')
    ma = tf.matmul(pa, a)
    mb = tf.matmul(pb, b)
    add_op = tf.add(ma, mb, name='Add')
    return pa, pb, add_op


def make_model_dyn_emb():
    a = de.get_variable(name='A', key_dtype=tf.int32, value_dtype=tf.int32, dim=8)
    b = tf.Variable(initial_value=np.ones([512, 8]), name='B', dtype=tf.int32)
    pa = tf.placeholder(dtype=tf.int32, shape=[None, 512], name='pa')
    pb = tf.placeholder(dtype=tf.int32, shape=[None, 512], name='pb')

    sp = tf.sparse.from_dense(pa)

    ma = de.embedding_lookup_sparse(a,
                                    tf.SparseTensor(indices=sp.indices, values=sp.values, dense_shape=sp.dense_shape),
                                    tf.SparseTensor(indices=sp.indices, values=sp.values, dense_shape=sp.dense_shape),
                                    )
    print('ma:', ma)
    mb = tf.matmul(pb, b)
    add_op = tf.add(ma, mb, name='Add')

    # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, a)

    return pa, pb, add_op
