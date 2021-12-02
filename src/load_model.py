import tensorflow._api.v2.compat.v1 as tf

tf.disable_eager_execution()

if __name__ == '__main__':
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['tag0'], "export_dir")
        graph = tf.get_default_graph()

        a = graph.get_tensor_by_name('a:0')
        b = graph.get_tensor_by_name('b:0')
        add_op = graph.get_tensor_by_name('add_a_b:0')
        ret = sess.run(add_op,
                       feed_dict={b: [2]})
        print("ret:", ret)
