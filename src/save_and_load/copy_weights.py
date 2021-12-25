import shutil

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import inner_export_utils
import export_paths

tf.disable_eager_execution()

base_dir = export_paths.base_dir
export_dir = export_paths.target_dir
migrate_dir = export_paths.migrate_dir

if __name__ == '__main__':
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], base_dir)
        # for c in tf.get_default_graph().get_operations():
        #     print(c)
        w_a = sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0])
        print('w_a:', w_a)

    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_dir)
        read_back = sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0])
        print('=================')
        print('before:', read_back)
        r_a = sess.graph.get_tensor_by_name('A:0')
        # for c in sess.graph.get_operations():
        #     print(c.name)

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
        inner_export_utils.write_graph(export_dir, sess.graph)
        # builder = tf.saved_model.builder.SavedModelBuilder(migrate_dir)
        # # builder.add_meta_graph_and_variables(sess,
        # #                                      meta_graph_def.meta_info_def.tags,
        # #                                      signature_def_map=meta_graph_def.signature_def
        # #                                      )
        #
        # collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # vl = list(filter(lambda x: x.name != 'avo', collection))
        # print(vl)
        # saver = tf.train.Saver(var_list=vl)
        # builder._has_saved_variables = True
        # builder.add_meta_graph(
        #     meta_graph_def.meta_info_def.tags,
        #     signature_def_map=meta_graph_def.signature_def,
        #     saver=saver
        # )
        # builder._has_saved_variables = False
        # builder.save()

    # with tf.Session(graph=tf.Graph()) as sess:
    #     tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], migrate_dir)
    #     tf.summary.FileWriter(migrate_dir, graph=sess.graph).flush()
    #     print('output read from revised graph:', sess.run(tf.get_default_graph().get_tensor_by_name("Add:0"),
    #                                                       feed_dict={'pa:0': np.repeat(1, 1024),
    #                                                                  'pb:0': np.repeat(1, 1024)}))
