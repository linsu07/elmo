import collections
import tensorflow as tf

tf.flags.DEFINE_integer('batch_size', 2, '一批数量样本的数量')
tf.flags.DEFINE_float("drop_out_rate", 0.5, "dropout概率，范围是0至1。例如rate=0.1会将输入Tensor的内容dropout10%。")
tf.flags.DEFINE_string("label_name", "label", "tfrecord中的标签的名字")
tf.flags.DEFINE_float("learning_rate", 0.01, '学习率.')
tf.flags.DEFINE_string("feature_name", "features", "tfrecord中的特征的名字")
tf.flags.DEFINE_string('log_level', 'INFO', 'tensorflow训练时的日志打印级别， 取值分别为，DEBUG，INFO,WARN,ERROR')
tf.flags.DEFINE_string('data_dir', 'D:/bilm-tf-master/data/elmo_pretreated.txt', '训练数据存放路径，支持hdfs')
tf.flags.DEFINE_string('model_dir', 'd:\\tmp\\elmo_model\\', '保存dnn模型文件的路径，支持hdfs')
tf.flags.DEFINE_list("gpu_cores",None,"例如[0,1,2,3]，在当个GPU机器的情况，使用的哪些核来训练")
tf.flags.DEFINE_integer("check_steps", 300,'保存训练中间结果的间隔，也是evalutation的间隔')
tf.flags.DEFINE_integer('max_steps', 1000, '训练模型最大的批训练次数，在model_dir不变的情况下重复训练')
tf.flags.DEFINE_integer("enable_ema",0,"是否启动指数移动平均来计算参数")
tf.flags.DEFINE_integer("cell_clip", 3, "梯度夹的大小，默认是3，一般不要超过5，避免梯度爆炸")
tf.flags.DEFINE_integer("projection_dim", 512, "投影层单元数")
tf.flags.DEFINE_integer("lstm_dim", 4096, "lstm单元个数")
tf.flags.DEFINE_integer("n_negative_samples_batch", 8192, "The number of classes to randomly sample per batch")
tf.flags.DEFINE_integer("n_tokens_vocab", 1000, "vocabulary 大小")
tf.flags.DEFINE_integer("unroll_steps", 20, "每条数据的步长")

class user_params(collections.namedtuple("namedtuple",
                                         ["label_name","learning_rate",
                                          "feature_name",
                                          "data_dir", "model_dir",
                                          "batch_size", "drop_out_rate",
                                          "enable_ema", "n_tokens_vocab",
                                          "projection_dim", "lstm_dim",
                                          "cell_clip", "n_negative_samples_batch",
                                          "unroll_steps"
                                          ])):
    pass

