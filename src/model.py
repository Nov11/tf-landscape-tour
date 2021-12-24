import tensorflow._api.v2.compat.v1 as tf


class SimpleModel:
    def __init__(self):
        self.w1 = tf.Variable(initial_value=tf.glorot_normal_initializer()(shape=[784, 512]), name='w1',
                              dtype=tf.float32)
        self.b1 = tf.Variable(initial_value=tf.glorot_normal_initializer()(shape=[512]), name='b1', dtype=tf.float32)

        self.w2 = tf.Variable(initial_value=tf.glorot_normal_initializer()(shape=[512, 10]), name='w2',
                              dtype=tf.float32)
        self.b2 = tf.Variable(initial_value=tf.glorot_normal_initializer()(shape=[10]), name='b2', dtype=tf.float32)

        # self.w3 = tf.Variable(initial_value=tf.random_normal([64, 10]), name='w3', dtype=tf.float32)
        # self.b3 = tf.Variable(initial_value=tf.random_normal([10]), name='b3', dtype=tf.float32)

        self.opt = tf.train.AdamOptimizer(0.001)

        self.feature = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='feature')
        self.label = tf.placeholder(dtype=tf.int64, shape=[None], name='label')

    def inference(self, data):
        o1 = tf.nn.relu(tf.matmul(data, self.w1, name="matmul_layer1") + self.b1, name='relu_layer1')
        return tf.matmul(o1, self.w2, name='matmul_layer2') + self.b2

    @staticmethod
    def compute_loss(raw_logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=raw_logits,
                                                                             name='sparse_softmax_cross_entrpy'),
                              name='loss_reduce_mean')

    def train(self, batch_feature, batch_label):
        with tf.GradientTape() as tape:
            raw_logits = self.inference(batch_feature)
            loss = self.compute_loss(raw_logits, batch_label)

        gradient = tape.gradient(target=loss, sources=tf.trainable_variables())
        return self.opt.apply_gradients(zip(gradient, tf.trainable_variables()),
                                        tf.train.get_or_create_global_step()), loss

    def evaluate(self, batch_feature, batch_label):
        raw = self.inference(batch_feature)
        return tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(raw, axis=1, name='argmax'), batch_label, name='acc_eq'), dtype=tf.float32,
                    name='acc_cast'), name='acc_reduce_mean')

    # def inference_pl(self):
    #     return self.inference(self.feature)

    def train_pl(self):
        return self.train(self.feature, self.label)

    def evaluate_pl(self):
        return self.evaluate(self.feature, self.label)

    def get_inputs(self):
        return self.feature, self.label
