import shutil

import tensorflow._api.v2.compat.v1 as tf

tf.disable_eager_execution()

base_dir = './base/'
export_dir = './target/'
migrate_dir = './migrate/'

if __name__ == '__main__':
    with tf.Session(graph=tf.Graph()) as sess:
        ret = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], base_dir)
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
        for c in sess.graph.get_operations():
            print(c.name)

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
        for c in sess.graph.get_operations():
            print(c.name)
        tf.summary.FileWriter(export_dir, graph=sess.graph).flush()

        shutil.rmtree(migrate_dir, ignore_errors=True)
        builder = tf.saved_model.builder.SavedModelBuilder(migrate_dir)
        # builder.add_meta_graph_and_variables(sess,
        #                                      meta_graph_def.meta_info_def.tags,
        #                                      signature_def_map=meta_graph_def.signature_def
        #                                      )

        collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vl = list(filter(lambda x: x.name != 'avo', collection))
        print(vl)
        saver = tf.train.Saver(var_list=vl)
        builder._has_saved_variables = True
        builder.add_meta_graph(
            meta_graph_def.meta_info_def.tags,
            signature_def_map=meta_graph_def.signature_def,
            saver=saver
        )
        builder._has_saved_variables = False
        builder.save()

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], migrate_dir)
        tf.summary.FileWriter(migrate_dir, graph=sess.graph).flush()
