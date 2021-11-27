""" prepare input dataset """

import logging

import tensorflow._api.v2.compat.v1 as tf
import tensorflow_datasets as tfds

logger = logging.getLogger(__name__)


def scale_param(image, label):
    return {"feature": tf.cast(tf.reshape(image, [-1]), tf.float32) / 255, "label": label}


def get_mnist_dataset():
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']
    logger.info("mnist_train %s", mnist_train)
    mnist_train = mnist_train.map(scale_param).shuffle(64).batch(64)
    logger.info("mnist_train scaled %s", mnist_train)
    mnist_test = mnist_test.map(scale_param).shuffle(64).batch(64)

    return mnist_train, mnist_test
