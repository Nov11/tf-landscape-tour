import logging

import tensorflow._api.v2.compat.v1 as tf

tf.disable_eager_execution()

logger = logging.getLogger(__name__)
"""not done yet"""
if __name__ == '__main__':
    saver = tf.train.import_meta_graph('./saved/mnist.ckpt.meta')
    with tf.Session() as sess:
        saver.restore(sess, './saved/mnist.ckpt')
        init_test_iter = tf.get_default_graph().get_operation_by_name("init_test_iter")
        logger.info("init test iter %s", sess.run(init_test_iter))

        acc_reduce_mean = tf.get_default_graph().get_tensor_by_name("acc_reduce_mean:0")
        logger.info("run test acc %s", sess.run(acc_reduce_mean))
