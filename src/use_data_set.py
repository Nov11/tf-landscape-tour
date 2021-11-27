import tensorflow._api.v2.compat.v1 as tf

tf.disable_eager_execution()

if __name__ == '__main__':
    ds = tf.data.Dataset.range(20).batch(2)

    print(ds)

    iterator = ds.make_initializable_iterator()
    get_next = iterator.train_get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)

        while True:
            try:
                print(sess.run(get_next))
            except tf.errors.OutOfRangeError:
                print("one epoch finished")
                break
