import tensorflow as tf
from ultra.elmo.network.elmo_head import ELMoHEAD
from ultra.elmo.network.language_model import language_model
from ultra.elmo.parameters import user_params
from ultra.elmo.network.embedding import embeddings


def model_fn(features: dict,labels,mode:tf.estimator.ModeKeys, params:user_params):
    with tf.variable_scope("elmo", initializer=tf.truncated_normal_initializer(mean=0.25, stddev=0.1)) as scope:

        features = embeddings(features, params, trainable=(mode==tf.estimator.ModeKeys.TRAIN))
        features = language_model(features, params, trainable=(mode==tf.estimator.ModeKeys.TRAIN))


    def train_op_fn(loss):
        global_step = tf.train.get_global_step()
        lr = tf.train.exponential_decay(params.learning_rate, global_step, 2000, 0.96, staircase=True)
        # lr = 0.001
        opt = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
        grads = opt.compute_gradients(loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
            gradients, 5)
        train_op = opt.apply_gradients(
            zip(capped_grads, variables), global_step=global_step)
        return train_op

    head = ELMoHEAD(params)

    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])

    for variable in variables:
        print(variable)

    return head.create_estimator_spec(features=features, mode=mode, logits=None, labels=labels, train_op_fn=train_op_fn)

