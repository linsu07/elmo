import sys
sys.path.insert(0, "/data/liutianyu/")
from tensorflow.python.estimator.run_config import RunConfig
import os
import pickle
from ultra.common.listeners import EvalListener
from ultra.elmo.input import DebugInput
from ultra.elmo.model import model_fn
from ultra.elmo.parameters import user_params
import tensorflow as tf
FLAGS = tf.flags.FLAGS

if __name__=="__main__":
    # 配置日志等级
    level_str = 'tf.logging.{}'.format(str(tf.flags.FLAGS.log_level).upper())
    tf.logging.set_verbosity(eval(level_str))

    FLAGS.n_tokens_vocab = 192798
    FLAGS.projection_dim = 512
    FLAGS.lstm_dim = 4096
    FLAGS.max_steps = 100000
    FLAGS.batch_size = 64
    FLAGS.n_negative_samples_batch = 8192
    FLAGS.data_dir = "/data/liutianyu/12elmo/data/train_elmo_estimator.txt"
    FLAGS.model_dir = "/data/liutianyu/12elmo/model/"
    FLAGS.check_steps = 500

    params = user_params(
        label_name=FLAGS.label_name,
        learning_rate=FLAGS.learning_rate,
        feature_name=FLAGS.feature_name,
        data_dir=FLAGS.data_dir,
        model_dir=FLAGS.model_dir,
        batch_size=FLAGS.batch_size,
        drop_out_rate=FLAGS.drop_out_rate,
        enable_ema=FLAGS.enable_ema,

        n_tokens_vocab=FLAGS.n_tokens_vocab,
        projection_dim=FLAGS.projection_dim,
        lstm_dim=FLAGS.lstm_dim,
        cell_clip=FLAGS.cell_clip,
        n_negative_samples_batch=FLAGS.n_negative_samples_batch,
        unroll_steps = FLAGS.unroll_steps
    )

    # 保存参数，方便其他调用elmo的网络加载
    with open(os.path.join(params.model_dir, "params.bin"), "wb") as f:
        pickle.dump(params, f)

    # 加载数据
    input_ = DebugInput(params)
    # input = SparkInput

    # tf.enable_eager_execution()
    # from tensorflow.contrib.eager.python.tfe import Iterator
    # for features, labels in Iterator(input_.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=params.data_dir)):
    #     spec = model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=params)
    #
    #     print("#########################")
    #     print(spec)
    #     break

    session_config = tf.ConfigProto()
    config = RunConfig(save_checkpoints_steps=FLAGS.check_steps, keep_checkpoint_max=2, session_config=session_config)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config, params=params)

    listeners = [
        EvalListener(estimator, lambda: input_.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=params.data_dir), name="train_data"),
        # VariableListener()
    ]

    def train_input_fn():
        return input_.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=params.data_dir)
    # estimator.train(train_input_fn, max_steps=FLAGS.max_steps, saving_listeners=listeners)
    estimator.train(train_input_fn, max_steps=FLAGS.max_steps)