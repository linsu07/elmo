# elmo预训练好的词向量模型，可以通过WarmStartSettings初始化其他模型的embedding层

import tensorflow as tf
from ultra.elmo.network.embedding import embeddings
from ultra.elmo.network.language_model import language_model
import pickle
import os

"""
用法说明
=======================================================
elmo_model_fn
应该在model_fn内部调用，当做embedding层
输入{"token_ids": Tensor[batch_size, sequence_length]}
返回 token_embedding, lstm_outputs_0, lstm_outputs_reverse_0, lstm_outputs_1, lstm_outputs_reverse_1他们都是Tensor[batch_size, sequence_length, embedding_size]

ELMoWarmStartHook
应该作为hook，传给estimator.train(input_fn=input_fn, max_steps=1000, hooks=[ELMoWarmStartHook(ckpt_path)])
输入elmo的checkpoint地址

elmo总体示例请见
ultra/elmo/warm_start_test.py
'''
def model_fn(features, labels, mode, params):

    to_elmo = {"token_ids": features["feature"]}

    elmo_result = elmo_model_fn(to_elmo, params["ckpt_path"])
    ...
    
ckpt_path = "/path/to/elmo/checkpoint"
estimator.train(input_fn=input_fn, max_steps=1000, hooks=[ELMoWarmStartHook(ckpt_path)])
'''
=======================================================
"""

def elmo_model_fn(features: dict,ckpt_path):
    """
    对输入的features["token_ids"] = [batch_size, sequence_length]
    输出elmo的embedding包括:
    token_embedding
    lstm_outputs_0: 第一层lstm正向embedding
    lstm_outputs_1: 第一层lstm正向embedding
    lstm_outputs_reverse_0: 第一层lstm正向embedding
    lstm_outputs_reverse_1: 第一层lstm正向embedding

    example:
    token_ids
    (?, 20)
    token_embedding
    (?, 20, 256)
    lstm_outputs_0
    (?, 20, 256)
    lstm_outputs_1
    (?, 20, 256)
    lstm_outputs_reverse_0
    (?, 20, 256)
    lstm_outputs_reverse_1
    (?, 20, 256)

    :param features: dict of tensor
    :param ckpt_path: elmo checkpoint path
    :return:
    """

    # 从elmo的checkpoint里，加载elmo的参数params
    params = None
    with open(os.path.join(ckpt_path, "params.bin"), "rb") as f:
        params = pickle.load(f)

    with tf.variable_scope("elmo", initializer=tf.truncated_normal_initializer(mean=0.25, stddev=0.1)) as scope:

        features = embeddings(features, params, trainable=False)
        features = language_model(features, params, is_predict=True, trainable=False)

    return features


class ELMoWarmStartHook(tf.train.SessionRunHook):
    def __init__(self, ckpt_path):
        self.ckpt_path=ckpt_path
    def begin(self):
        assignment_map = {
            "elmo/embeddings/embedding":"elmo/embeddings/embedding",
            'elmo/language_model/RNN_fw_0/rnn/lstm_cell/bias':"elmo/language_model/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias",
            'elmo/language_model/RNN_fw_0/rnn/lstm_cell/kernel':"elmo/language_model/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel",
            'elmo/language_model/RNN_fw_0/rnn/lstm_cell/projection/kernel':"elmo/language_model/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/projection/kernel",
            'elmo/language_model/RNN_bw_0/rnn/lstm_cell/bias':"elmo/language_model/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/bias",
            'elmo/language_model/RNN_bw_0/rnn/lstm_cell/kernel':"elmo/language_model/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel",
            'elmo/language_model/RNN_bw_0/rnn/lstm_cell/projection/kernel':"elmo/language_model/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/projection/kernel",
            'elmo/language_model/RNN_fw_1/rnn/lstm_cell/bias':"elmo/language_model/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/bias",
            'elmo/language_model/RNN_fw_1/rnn/lstm_cell/kernel':"elmo/language_model/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel",
            'elmo/language_model/RNN_fw_1/rnn/lstm_cell/projection/kernel':"elmo/language_model/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/projection/kernel",
            'elmo/language_model/RNN_bw_1/rnn/lstm_cell/bias':"elmo/language_model/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/bias",
            'elmo/language_model/RNN_bw_1/rnn/lstm_cell/kernel':"elmo/language_model/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel",
            'elmo/language_model/RNN_bw_1/rnn/lstm_cell/projection/kernel':"elmo/language_model/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/projection/kernel"
        }

        tf.train.init_from_checkpoint(self.ckpt_path, {v:k for k, v in assignment_map.items()})