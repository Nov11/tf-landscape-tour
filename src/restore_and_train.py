import logging

import tensorflow._api.v2.compat.v1 as tf

from src import mnist_data

tf.disable_eager_execution()

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    _, mnist_test = mnist_data.get_mnist_dataset()

    ds_iter = tf.data.make_initializable_iterator(mnist_test)

    ds_get_next = ds_iter.get_next(name='test_iter_get_next')

    saver = tf.train.import_meta_graph('./saved/mnist.ckpt.meta', import_scope='new_scope')
    with tf.Session() as sess:
        saver.restore(sess, './saved/mnist.ckpt')

        sess.run(ds_iter.initializer)

        init_test_iter = tf.get_default_graph().get_tensor_by_name("test_iter_get_next")
        logger.info("init test iter %s", sess.run(init_test_iter))

        acc_reduce_mean = tf.get_default_graph().get_tensor_by_name("acc_reduce_mean:0")
        logger.info("run test acc %s", sess.run(acc_reduce_mean))



