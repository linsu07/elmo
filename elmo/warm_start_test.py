import sys
sys.path.insert(0, "/data/liutianyu/")
from tensorflow.contrib.estimator import multi_class_head
import tensorflow as tf
from ultra.elmo.warm_start_handler import elmo_model_fn, ELMoWarmStartHook

batch_size = 2

def model_fn(features, labels, mode, params):

    to_elmo = {"token_ids": features["feature"]}

    elmo_result = elmo_model_fn(to_elmo, params["ckpt_path"])

    feature = tf.add_n(
        [elmo_result["token_embedding"],
        elmo_result["lstm_outputs_0"],
        elmo_result["lstm_outputs_reverse_0"],
        elmo_result["lstm_outputs_1"],
        elmo_result["lstm_outputs_reverse_1"]]
    )
    logits = tf.reduce_mean(feature, axis=-2)

    logits = tf.layers.dense(logits, 3)

    def train_op_fn(loss):
        global_step=tf.train.get_global_step()
        opt = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
        grads = opt.compute_gradients(loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
            gradients, 5)
        train_op = opt.apply_gradients(
            zip(capped_grads, variables), global_step=global_step)
        return train_op

    head = multi_class_head(3)
    spec = head.create_estimator_spec(features, mode, logits, labels=labels, train_op_fn=train_op_fn, )

    # variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    # for variable in variables:
    #     print(variable)

    return spec


def input_fn():
    feature = [[52, 60, 56, 27, 358, 74, 30, 63, 64, 117, 41, 28, 2462, 3342, 3286, 2063, 2066, 26, 2125, 138],
               [594, 41, 9449, 2556, 85, 106, 1915, 51, 1, 451, 330, 1338, 41, 786, 2289, 237, 373, 243, 2289, 86]]
    label = [[1], [0]]
    dataset = tf.data.Dataset.from_tensor_slices(({"feature":feature}, label))
    return dataset.repeat(None).batch(batch_size)



if __name__ == "__main__":

    tf.logging.set_verbosity("DEBUG")
    tf.set_random_seed(666)
    ckpt_path = "/data/liutianyu/12elmo/model"
    estimator = tf.estimator.Estimator(model_fn=model_fn, params={"ckpt_path": ckpt_path, "batch_size": batch_size})
    estimator.train(input_fn=input_fn, max_steps=1000, hooks=[ELMoWarmStartHook(ckpt_path)])
    # estimator.train(input_fn=input_fn, max_steps=1000)

