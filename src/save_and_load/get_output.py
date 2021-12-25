import numpy as np
import setuptools.glob
import tensorflow._api.v2.compat.v1 as tf

import export_paths

base_dir = export_paths.base_dir
export_dir = export_paths.target_dir
migrate_dir = export_paths.migrate_dir

tf.disable_eager_execution()

# if __name__ == '__main__':
# with tf.Session(graph=tf.Graph()) as sess:
#     tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], migrate_dir)
#     tf.summary.FileWriter(migrate_dir, graph=sess.graph).flush()
#     print('output read from revised graph:', sess.run(tf.get_default_graph().get_tensor_by_name("Add:0"),
#                                                       feed_dict={'pa:0': np.repeat(1, 1024),
#                                                                  'pb:0': np.repeat(1, 1024)}))

if __name__ == '__main__':
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], base_dir)
        w_a = sess.run(sess.graph.get_operation_by_name('A/Read/ReadVariableOp').outputs[0])
        print('w_a:', w_a)
        add = sess.run(sess.graph.get_tensor_by_name('Add:0'), feed_dict={
            sess.graph.get_tensor_by_name('pa:0'): np.ones([1024]),
            sess.graph.get_tensor_by_name('pb:0'): np.ones([1024]),

        })
        print('add:', add)
