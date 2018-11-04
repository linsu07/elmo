import sys
sys.path.insert(0, "/data/liutianyu/ultra/elmo")

from tensorflow.python.estimator.run_config import RunConfig, TaskType
import pickle
from ultra.common import MyTraining
from ultra.common.listeners import LoadEMAHook, EvalListener
from ultra.elmo.input import SparkInput, DebugInput
from ultra.elmo.model import model_fn
from ultra.elmo.parameters import user_params
import os
import tensorflow as tf
FLAGS = tf.flags.FLAGS

def train(params: user_params, input_):
    # estimator运行环境配置
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session_config.gpu_options.allow_growth = True

    if FLAGS.gpu_cores:
        gpu_cors = tuple(FLAGS.gpu_cores)
        devices =  ["/device:GPU:%d" % int(d) for d in gpu_cors]
        tf.logging.warn("using device: " + " ".join(devices))
        distribution = tf.contrib.distribute.MirroredStrategy(devices = devices)

        tf.logging.warn("in train.py, distribution")
        tf.logging.warn(distribution._devices)

        config = RunConfig(save_checkpoints_steps=FLAGS.check_steps,train_distribute=distribution, keep_checkpoint_max=2, session_config=session_config)
    else:

        config = RunConfig(save_checkpoints_steps=FLAGS.check_steps, keep_checkpoint_max=2, session_config=session_config)


    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config, params=params)

    train_data_dir = input_.get_data_dir(tf.estimator.ModeKeys.TRAIN, params)
    eval_data_dir = input_.get_data_dir(tf.estimator.ModeKeys.EVAL, params)

    hook = [] if not params.enable_ema else [LoadEMAHook(params.model_dir,0.99)]

    listeners = [
        EvalListener(estimator, lambda: input_.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=train_data_dir), name="train_data",hook = hook),
        EvalListener(estimator, lambda: input_.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=eval_data_dir),hook = hook),
        # VariableListener()
    ]

    def train_input_fn():
        return input_.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=train_data_dir)

    # gpu cluster
    if config.cluster_spec:
        train_spec = MyTraining.TrainSpec(train_input_fn, FLAGS.max_steps)
        eval_spec = MyTraining.EvalSpec(lambda: input_.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=train_data_dir), steps=FLAGS.check_steps)
        MyTraining.train_and_evaluate(estimator, train_spec, eval_spec, listeners)
        if config.task_type == TaskType.CHIEF:
            model_dir = estimator.export_savedmodel(FLAGS.model_dir, input_.get_input_reciever_fn())
            tf.logging.warn("save model to %s" % model_dir)

    # cpu solo
    else:
        # from tensorflow.python import debug as tf_debug
        # debug_hook = [tf_debug.LocalCLIDebugHook(ui_type="readline")]
        # estimator.train(train_input_fn, max_steps=FLAGS.max_steps, saving_listeners=listeners, hooks=debug_hook)
        estimator.train(train_input_fn, max_steps=FLAGS.max_steps, saving_listeners=listeners)
        dir = estimator.export_savedmodel(tf.flags.FLAGS.model_dir, input_.get_input_reciever_fn())
        tf.logging.warn("save model to %s" % dir)

    for listener in listeners:
        print(listener.name)
        print(listener.history)

def main(_):

    # 配置日志等级
    level_str = 'tf.logging.{}'.format(str(tf.flags.FLAGS.log_level).upper())
    tf.logging.set_verbosity(eval(level_str))

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
    # input = SparkInput
    input_ = DebugInput(params)

    # tf.enable_eager_execution()
    # from tensorflow.contrib.eager.python.tfe import Iterator
    # for item in Iterator(input_.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=params.data_dir)):
    #     print(item[0])
    #     print(item[1])
    #     print("#########################")
    #     break

    train(params, input_)



if __name__ == "__main__":
    tf.app.run(main)