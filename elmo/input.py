import tensorflow as tf
import os
from elmo.parameters import user_params
import numpy as np
import json

class BaseInput():
    def __init__(self, params: user_params):
        self.feature_name = params.feature_name
        self.conttext_spec = {}
        self.conttext_spec[params.label_name] = tf.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0)
        self.sequence_spec = {
            params.feature_name:tf.VarLenFeature(tf.float32),
        }

    def get_data_dir(self, mode: tf.estimator.ModeKeys, params: user_params):
        return os.path.join(params.data_dir, "train") if mode == tf.estimator.ModeKeys.TRAIN else os.path.join(params.data_dir, "evaluation")
    def input_fn(self, mode: tf.estimator.ModeKeys, params: user_params, data_dir):pass
    def get_input_reciever_fn(self):pass

class SparkInput(BaseInput):
    pass


class DebugInput(BaseInput):

    def input_fn(self, mode: tf.estimator.ModeKeys, params: user_params, data_dir):
        print(data_dir)
        file_paths = tf.gfile.Glob(data_dir)

        dataset = tf.data.TextLineDataset(file_paths)

        def function(raws):
            d = json.loads(raws.decode('utf-8'))
            return np.int32(d["token_ids"]), np.int32(d["token_ids_reverse"]), np.int32(d["next_token_id"]), np.int32(d["next_token_id_reverse"])


        if mode == tf.estimator.ModeKeys.TRAIN:
            data_set = dataset.repeat(None).shuffle(buffer_size=20) \
                .map(lambda x: tf.py_func(function, [x], Tout=[tf.int32, tf.int32, tf.int32, tf.int32])) \
                .map(
                lambda v,x,y,z: (
                    {
                        "token_ids":tf.reshape(v, [params.unroll_steps]),
                        "token_ids_reverse":tf.reshape(x, [params.unroll_steps])
                    }, {
                        "next_token_id":tf.reshape(y, [params.unroll_steps]),
                        "next_token_id_reverse":tf.reshape(z, [params.unroll_steps])})) \
                .batch(params.batch_size)

        elif mode == tf.estimator.ModeKeys.EVAL:
            data_set = dataset.repeat(1) \
                .map(lambda x: tf.py_func(function, [x], Tout=[tf.int32, tf.int32, tf.int32, tf.int32])) \
                .map(lambda v,x,y,z: ({"token_ids":v, "token_ids_reverse":x}, {"next_token_id":y, "next_token_id_reverse":z})) \
                .batch(params.batch_size).take(5)
        return data_set
    def get_input_reciever_fn(self):pass