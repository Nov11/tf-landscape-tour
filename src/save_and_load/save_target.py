import export_paths
import inner_export_utils
import tensorflow._api.v2.compat.v1 as tf

tf.disable_eager_execution()

export_dir = export_paths.target_dir

if __name__ == '__main__':
    inner_export_utils.clean_directories(export_dir)

    with tf.Graph().as_default() as g:
        a = tf.Variable(initial_value=[100], name='A')
        b = tf.Variable(initial_value=[4], name='B')
        pa = tf.placeholder(dtype=tf.int32, shape=[1], name='pa')
        pb = tf.placeholder(dtype=tf.int32, shape=[1], name='pb')
        ma = tf.multiply(a, pa)
        mb = tf.multiply(b, pb)
        add_op = tf.add(ma, mb, name='Add')

        with tf.Session(graph=g) as sess:
            tf.global_variables_initializer().run()

            inner_export_utils.do_save_model(export_dir, sess, pa, pb, add_op)
            inner_export_utils.write_graph(export_dir, sess.graph)
