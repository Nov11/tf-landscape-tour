import shutil

import tensorflow._api.v2.compat.v1 as tf
import export_paths
import inner_export_utils

from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants

tf.disable_eager_execution()

base_dir = export_paths.base_dir
target_dir = export_paths.target_dir
migrate_dir = export_paths.migrate_dir
freezed_dir = export_paths.freezed_dir

if __name__ == '__main__':
    # load weights from base model
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], base_dir)
        w_a = sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0])
        print('w_a:', w_a)

    # load target model and update weights of A
    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], target_dir)
        print('=================before:',
              sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0]))

        sess.run(tf.raw_ops.AssignVariableOp(resource=sess.graph.get_tensor_by_name('A:0'), value=w_a, name='avo'))
        print('=================after:',
              sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0]))

        tf.summary.FileWriter(target_dir + '/graph/', graph=sess.graph).flush()

        shutil.rmtree(freezed_dir, ignore_errors=True)

        output_graph_def = convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names=['Add'])
        tf.train.write_graph(output_graph_def, freezed_dir, 'freeze.pb', as_text=False)

    shutil.rmtree(migrate_dir, ignore_errors=True)
    with tf.Session(graph=tf.Graph()) as sess:
        d = tf.GraphDef()
        with open(freezed_dir + '/freeze.pb', 'rb') as f:
            d.ParseFromString(f.read())
            tf.import_graph_def(d, name='')

        print('=================load from freeze:', sess.run(sess.graph.get_operation_by_name('A').outputs[0]))
        for c in tf.get_default_graph().get_operations():
            print(c.name)

        pa = sess.graph.get_tensor_by_name("pa:0")
        pb = sess.graph.get_tensor_by_name("pb:0")
        out = sess.graph.get_tensor_by_name("Add:0")

        print('=================freeze to saved model')
        inner_export_utils.do_save_model(migrate_dir, sess, pa, pb, out)
        inner_export_utils.write_graph(migrate_dir, sess.graph)

    with tf.Session(graph=tf.Graph()) as sess:
        print('=================check loading saved model')
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], migrate_dir)
        print('=================check load from freeze:', sess.run(sess.graph.get_operation_by_name('A').outputs[0]))
        for c in tf.get_default_graph().get_operations():
            print(c.name)
        inner_export_utils.write_graph(migrate_dir + '/from_saved_model', sess.graph)
