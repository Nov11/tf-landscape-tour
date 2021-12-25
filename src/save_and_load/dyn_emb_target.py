import shutil

import tensorflow._api.v2.compat.v1 as tf
import export_paths
import tensorflow_recommenders_addons.dynamic_embedding as de

tf.disable_eager_execution()

export_dir = export_paths.dyn_emb_dir

if __name__ == '__main__':
    shutil.rmtree(export_dir, ignore_errors=True)

    a = de.get_variable(name='A')
    b = tf.Variable(initial_value=[4], name='B')
    add_op = tf.add(a, b)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.TRAINING],
                                             signature_def_map=
                                             {'default':
                                                 tf.saved_model.signature_def_utils.build_signature_def(
                                                     inputs={
                                                         't_ia': tf.saved_model.utils.build_tensor_info(a),
                                                         't_ib': tf.saved_model.utils.build_tensor_info(b)
                                                     },
                                                     outputs={
                                                         't_output': tf.saved_model.utils.build_tensor_info(add_op)
                                                     },
                                                     method_name='t_operation_add'
                                                 )
                                             },
                                             )
        builder.save()
        writer = tf.summary.FileWriter(export_dir, graph=sess.graph)
        writer.flush()
        writer.close()
