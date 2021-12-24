import shutil

import tensorflow._api.v2.compat.v1 as tf
import export_paths

tf.disable_eager_execution()

base_dir = export_paths.base_dir

if __name__ == '__main__':
    a = tf.Variable(initial_value=[2], name='A')
    b = tf.Variable(initial_value=[4], name='B')
    add_op = tf.add(a, b)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        shutil.rmtree(base_dir, ignore_errors=True)
        builder = tf.saved_model.builder.SavedModelBuilder(base_dir)
        # g = tf.Graph()
        # with g.as_default():
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.TRAINING],
                                             signature_def_map=
                                             {'default':
                                                 tf.saved_model.signature_def_utils.build_signature_def(
                                                     inputs={
                                                         'ia': tf.saved_model.utils.build_tensor_info(a),
                                                         'ib': tf.saved_model.utils.build_tensor_info(b)
                                                     },
                                                     outputs={
                                                         'output': tf.saved_model.utils.build_tensor_info(add_op)
                                                     },
                                                     method_name='operation_add'
                                                 )
                                             },
                                             )
        builder.save()
        writer = tf.summary.FileWriter(base_dir, graph=sess.graph)
        writer.flush()
        writer.close()
