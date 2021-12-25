import shutil

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import inner_export_utils
import export_paths

tf.disable_eager_execution()

base_dir = export_paths.base_dir
target_dir = export_paths.dyn_emb_dir
migrate_dir = export_paths.dyn_emb_out_dir

if __name__ == '__main__':
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], base_dir)
        inner_export_utils.show_op_names()
        w_a = sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0])
        print('w_a:', w_a)

    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], target_dir)
        inner_export_utils.show_op_names()
        print("dyn emb model loaded")

        read_back = sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0])
        print('=================')
        print('before:', read_back)
        r_a = sess.graph.get_tensor_by_name('A:0')

        # placeholder = tf.placeholder(dtype=r_a.dtype, shape=r_a.shape)
        # sess.run(tf.raw_ops.AssignVariableOp(resource=r_a, value=placeholder), feed_dict={placeholder: w_a})
        sess.run(tf.raw_ops.AssignVariableOp(resource=r_a, value=w_a, name='avo'))
        # with tf.Graph().as_default() as g:
        #     print('=================assign')
        #     for c in g.get_operations():
        #         print(c.name)
        read_back = sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0])
        print('=================')
        print('after:', read_back)
        # for c in sess.graph.get_operations():
        #     print(c.name)

        shutil.rmtree(migrate_dir, ignore_errors=True)

        pa = sess.graph.get_tensor_by_name("pa:0")
        pb = sess.graph.get_tensor_by_name("pb:0")
        out = sess.graph.get_tensor_by_name("Add:0")
        inner_export_utils.do_save_model(migrate_dir, sess, pa, pb, out)
        inner_export_utils.write_graph(target_dir, sess.graph)
