import numpy as np
import tensorflow._api.v2.compat.v1 as tf

import export_paths
import inner_export_utils
import simple_model

# import tensorflow_recommenders_addons.dynamic_embedding as de

tf.disable_eager_execution()

if __name__ == '__main__':
    # not working :KeyError: "The name 'A' refers to an Operation not in the graph."
    # with tf.Session(graph=tf.Graph()) as sess:
    #     meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING],
    #                                                 export_paths.dyn_emb_dir)
    #     inner_export_utils.show_op_names()

    with tf.Session(graph=tf.Graph()) as sess:
        pa, pb, o = simple_model.make_model_dyn_emb()
        saver = tf.train.Saver()
        saver.restore(sess, export_paths.dyn_emb_dir + '/ckpt')
        print(sess.run(o, feed_dict={pa: np.ones([1, 512]), pb: np.ones([1, 512])}))
        inner_export_utils.show_op_names()
