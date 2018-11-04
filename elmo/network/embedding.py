from ultra.elmo.parameters import user_params
import tensorflow as tf

class Embeddings(tf.layers.Layer):
    def __init__(self, params: user_params, trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(Embeddings, self).__init__(trainable=trainable, name=name,
                                      activity_regularizer=activity_regularizer,
                                      **kwargs)
        self.params = params

    def build(self, input_shape):
        self.embedding_weights = tf.get_variable(
            "embedding", [self.params.n_tokens_vocab, self.params.projection_dim],
            dtype=tf.float32,
            trainable=self.trainable
        )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        """
        embedding_loopup
        :param inputs: 至少包含token_ids单方向，训练时应该还包含了token_ids_reverse反向。训练时，两个方向都是随机抽取的，内容不同。
        :param args:
        :param kwargs:
        :return:
        """

        # the input token_ids and word embeddings
        inputs["token_embedding"] = tf.nn.embedding_lookup(self.embedding_weights, inputs["token_ids"], name="token_embedding")
        inputs["seq_length"] = tf.cast(tf.reduce_sum(tf.sign(inputs["token_ids"]), axis=-1), tf.int32)
        # if a bidirectional LM then make placeholders for reverse
        # model and embeddings
        token_ids_reverse = inputs.get("token_ids_reverse", None)
        if token_ids_reverse != None:
            inputs["token_reverse_embedding"] = tf.nn.embedding_lookup(self.embedding_weights, token_ids_reverse, name="token_reverse_embedding")

        return inputs

    def __call__(self, inputs, *args, **kwargs):
        return super(Embeddings, self).__call__(inputs, *args, **kwargs)

def embeddings(inputs, params: user_params, trainable=True, name=None, dtype=None,
               activity_regularizer=None, **kwargs):
    layer = Embeddings(params, trainable, name, dtype,
                    activity_regularizer, **kwargs)
    return layer(inputs)


# def build_word_embeddings(features, params: user_params):
#
#     # the input token_ids and word embeddings
#     token_ids = features["token_ids"]
#
#     # the word embeddings
#     embedding_weights = tf.get_variable(
#         "embedding", [params.n_tokens_vocab, params.projection_dim],
#         dtype=tf.float32,
#     )
#     token_embedding = tf.nn.embedding_lookup(embedding_weights, token_ids, name="token_embedding")
#
#     # if a bidirectional LM then make placeholders for reverse
#     # model and embeddings
#     token_ids_reverse = features["token_ids_reverse"]
#     token_reverse_embedding = tf.nn.embedding_lookup(embedding_weights, token_ids_reverse, name="token_reverse_embedding")
#
#     features["token_embedding"] = token_embedding
#     features["token_reverse_embedding"] = token_reverse_embedding
#     return features