import os.path
import shutil

import tensorflow._api.v2.compat.v1 as tf

tf.disable_eager_execution()

if __name__ == '__main__':
    ckpt = 'find_saved_var'
    ex = 'export_dir'
    shutil.rmtree(ckpt, ignore_errors=True)
    shutil.rmtree(ex, ignore_errors=True)

    a = tf.Variable([1], dtype=tf.int64, name='a')
    b = tf.placeholder(dtype=tf.int64, name='b')

    add_op = tf.add(a, b, name='add_a_b')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(add_op, feed_dict={
            a: [1],
            b: [2]
        }))
        saver = tf.train.Saver()
        saver.save(sess, ckpt)
        builder = tf.saved_model.Builder(ex)
        builder.add_meta_graph_and_variables(sess,
                                             tags=['tag0'],
                                             signature_def_map=
                                             {
                                                 tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                     tf.saved_model.signature_def_utils.build_signature_def(
                                                         inputs={
                                                             "va": tf.saved_model.utils.build_tensor_info(a),
                                                             "vb": tf.saved_model.utils.build_tensor_info(b)
                                                         },
                                                         outputs={
                                                             "add_result": tf.saved_model.utils.build_tensor_info(
                                                                 add_op)
                                                         },
                                                         method_name='add_a_and_b'
                                                     )},
                                             saver=saver
                                             )
        builder.save()
