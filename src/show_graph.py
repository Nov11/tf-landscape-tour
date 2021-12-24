import tensorflow._api.v2.compat.v1 as tf

tf.disable_eager_execution()

if __name__ == '__main__':
    a = tf.constant([1], name='a')
    b = tf.constant([2], name='b')

    add_op = tf.add(a, b, name='add_a_b')

    writer = tf.summary.FileWriter("logdir/show_graph")
    with tf.Session() as sess:
        writer.add_graph(sess.graph)

        ret = sess.run(add_op)
        print(ret)

    writer.flush()
