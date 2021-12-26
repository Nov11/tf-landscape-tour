import logging
import shutil

import numpy as np
import tensorflow._api.v2.compat.v1 as tf

import export_paths
import inner_export_utils
import simple_model

import tensorflow_recommenders_addons.dynamic_embedding as de

tf.disable_eager_execution()

load_dir = export_paths.dyn_emb_dir
output_dir = export_paths.dyn_emb_out_dir

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # not working :KeyError: "The name 'A' refers to an Operation not in the graph."
    # with tf.Session(graph=tf.Graph()) as sess:
    #     meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING],
    #                                                 export_paths.dyn_emb_dir)
    #     inner_export_utils.show_op_names()

    new_key = np.arange(20, 30)
    new_val = np.tile(np.arange(0, 8), [10, 1])
    with tf.Session(graph=tf.Graph()) as sess:
        pa, pb, o = simple_model.make_model_dyn_emb()
        saver = tf.train.Saver()
        saver.restore(sess, load_dir + '/ckpt')
        logger.info(sess.run(o, feed_dict={pa: np.ones([1, 512]), pb: np.ones([1, 512])}))
        # inner_export_utils.show_op_names()
        # this is not applicable
        # w_a = sess.run('A:0')
        # print(w_a)

        logger.info("assign weights")
        with tf.variable_scope(name_or_scope='', reuse=True):
            v: de.Variable = de.get_variable('A')
            logger.info(sess.run(v.export("export_values0")))

            sess.run(v.upsert(new_key, new_val, name='upsert'))
            logger.info(sess.run(v.export("export_values1")))

        shutil.rmtree(output_dir, ignore_errors=True)
        inner_export_utils.do_save_model(output_dir, sess, pa, pb, o)
        inner_export_utils.write_graph(output_dir, sess.graph)
        saver = tf.train.Saver()
        saver.save(sess, output_dir + '/ckpt')

    with tf.Session(graph=tf.Graph()) as sess:
        pa, pb, o = simple_model.make_model_dyn_emb()
        saver = tf.train.Saver()
        saver.restore(sess, output_dir + '/ckpt')
        logger.info(sess.run(o, feed_dict={pa: np.ones([1, 512]), pb: np.ones([1, 512])}))

        with tf.variable_scope(name_or_scope='', reuse=True):
            v: de.Variable = de.get_variable('A')
            logger.info(sess.run(v.export("export_values0")))