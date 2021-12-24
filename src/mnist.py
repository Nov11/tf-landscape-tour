import logging

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import tensorflow_datasets as tfds

from src import mnist_data
from src.model import SimpleModel

tf.disable_eager_execution()

logger = logging.getLogger(__name__)


def get_dataset_builder(name):
    mnist_builder = tfds.builder(name)
    mnist_builder.download_and_prepare()
    builder = mnist_builder.as_dataset(split="train")
    print(builder.info)
    print(builder.info.features)
    print(builder.info.features["label"].num_classes)
    print(builder.info.features["label"].names)
    return builder


if __name__ == '__main__':
    mnist_train, mnist_test = mnist_data.get_mnist_dataset()

    train_iter = tf.data.make_initializable_iterator(mnist_train)
    train_iter_init = train_iter.make_initializer(mnist_train, name="init_train_iter")
    train_get_next = train_iter.get_next(name='train_iter_get_next')

    test_iter = tf.data.make_initializable_iterator(mnist_test)
    test_iter_init = test_iter.make_initializer(mnist_test, name='init_test_iter')
    test_get_next = test_iter.get_next(name='test_iter_get_next')

    model = SimpleModel()
    in_feature, in_label = model.get_inputs()
    _, _ = model.train_pl()
    opt, loss = model.train(train_get_next['feature'], train_get_next['label'])
    _ = model.evaluate_pl()
    acc = model.evaluate(test_get_next['feature'], test_get_next['label'])

    global_step = tf.train.get_or_create_global_step()
    sum_loss_op = tf.summary.scalar('loss', loss)
    sum_acc_op = tf.summary.scalar('acc', acc)

    with tf.Session() as sess, tf.summary.FileWriter(logdir='logdir', graph=sess.graph) as writer:
        sess.run(tf.global_variables_initializer())

        for i in range(2):
            # logger.info("global step %s", sess.run(global_step))
            sess.run(train_iter_init)
            train_loss = np.array([])
            while True:
                try:
                    _, ret_loss, ret_sum_loss, step = sess.run([opt, loss, sum_loss_op, global_step])
                    train_loss = np.append(train_loss, [ret_loss])
                    writer.add_summary(ret_sum_loss, step)
                except tf.errors.OutOfRangeError:
                    break

            sess.run(test_iter_init)
            test_acc = np.array([])
            while True:
                try:
                    ret, ret_sum_acc, step = sess.run([acc, sum_acc_op, global_step])
                    writer.add_summary(ret_sum_acc, step)
                    test_acc = np.append(test_acc, [ret])
                except tf.errors.OutOfRangeError:
                    break

            logger.info("epoch[%s] loss:%s acc: %s", i, np.mean(train_loss), np.mean(test_acc))

        writer.flush()
        saver = tf.train.Saver()
        saver.save(sess, './saved/mnist.ckpt')
