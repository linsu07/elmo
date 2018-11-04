import tensorflow as tf
from elmo.parameters import user_params

class LanguageModel(tf.layers.Layer):
    def __init__(self, params: user_params, is_predict:bool=False, trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(LanguageModel, self).__init__(trainable=trainable, name=name,
                                         activity_regularizer=activity_regularizer,
                                         **kwargs)
        self.params = params
        self.is_predict = is_predict
        self.cell_dict = {}
        self.directions = ["fw", "bw"]

    def build_one_direction(self, name):
        """
        构建2层lstm encoder
        :return: MultiRNNCell对象
        """
        lstm_cells = []
        # 2 layer lstm
        for i in range(2):
            if self.params.projection_dim < self.params.lstm_dim:
                # are projecting down output
                lstm_cell = tf.nn.rnn_cell.LSTMCell(
                    self.params.lstm_dim, num_proj=self.params.projection_dim,
                    cell_clip=self.params.cell_clip, proj_clip=self.params.cell_clip)
                lstm_cell.trainable = self.trainable
            else:
                lstm_cell = tf.nn.rnn_cell.LSTMCell(
                    self.params.lstm_dim,
                    cell_clip=self.params.cell_clip, proj_clip=self.params.cell_clip)
                lstm_cell.trainable = self.trainable

                # ResidualWrapper adds inputs to outputs
            if i != 0:
                # don't add skip connection from token embedding to
                # 1st layer output
                lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # add dropout
            if self.trainable:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1-self.params.drop_out_rate)

            lstm_cells.append(lstm_cell)
            self.cell_dict[name + "_" + str(i)] = lstm_cell

        # MultiRNNCell，便于多层rnn合一cell
        return tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

    def build(self, input_shape):
        """
        输入中包含了token/token_reverse，2套embedding
        分别构建2层lstm encoder
        :param input_shape:
        :return:
        """
        self.two_direction_cell = []

        for direction in self.directions:

            lstm_cell = self.build_one_direction(direction)
            self.two_direction_cell.append(lstm_cell)

        self.built = True

    def call(self, inputs, *args, **kwargs):
        """
        is_predict，决定了elmo工作在独自训练过程，还是集成在其他神经网络中作为embedding层
        :param inputs:
        :param args:
        :param kwargs:
        :return:
        """

        # 先get正向token
        token_embedding = inputs["token_embedding"]

        # 再get反向token
        # 训练时，一定存在反向token
        # 预测时，可能不存在，所以需要reverse_sequence来得到
        token_reverse_embedding = inputs.get("token_reverse_embedding", None)
        seq_length = inputs["seq_length"]
        if token_reverse_embedding == None:
            token_reverse_embedding = tf.reverse_sequence(token_embedding, seq_lengths=seq_length, seq_axis=1, batch_axis=0)

        # get the LSTM inputs
        lstm_inputs = [token_embedding, token_reverse_embedding]

        init_lstm_state = []
        final_lstm_state = []
        batch_size = tf.shape(inputs["token_embedding"])[0]

        # 这里有所区别
        # 预测过程中，正反向token原于预测文本，batch内每条长度不同，固需要dynamic_rnn
        if self.is_predict:
            # 2个方向
            for direction_id,  direction in enumerate(self.directions):
                layer_input = lstm_inputs[direction_id]
                # 2层lstm
                for layer in range(2):
                    cell_name = direction + "_" + str(layer)
                    lstm_cell = self.cell_dict.get(cell_name)

                    with tf.variable_scope('RNN_' + cell_name):
                        layer_output, final_state = tf.nn.dynamic_rnn(
                            lstm_cell,
                            layer_input,
                            sequence_length=seq_length,
                            # initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                            #     *batch_init_states)
                            dtype=tf.float32
                        )
                    if direction_id == 1:
                        # reverse方向，输出前翻转回正向
                        inputs["lstm_outputs_reverse_" + str(layer)] = tf.reverse_sequence(
                            layer_output,
                            seq_lengths=seq_length,
                            seq_axis=1,
                            batch_axis=0
                        )
                    else:
                        inputs["lstm_outputs_" + str(layer)] = layer_output
                    layer_input = layer_output

        # 训练过程中，正反向token虽然是随机取自训练文本，但长度固定是20，固可以用static_rnn
        else:
            for lstm_num, lstm_input in enumerate(lstm_inputs):
                lstm_cell = self.two_direction_cell[lstm_num]
                with tf.control_dependencies([lstm_input]):
                    init_lstm_state.append(
                        lstm_cell.zero_state(batch_size, tf.float32))

                    with tf.variable_scope('RNN_%s' % lstm_num):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            lstm_cell,
                            tf.unstack(lstm_input, num=self.params.unroll_steps, axis=1),
                            initial_state=init_lstm_state[-1])
                    final_lstm_state.append(final_state)


                # (batch_size * unroll_steps, projection_dim)
                lstm_output_flat = tf.reshape(
                    tf.stack(_lstm_output_unpacked, axis=1), [-1, self.params.projection_dim])
                if self.trainable:
                    # add dropout to output
                    lstm_output_flat = tf.nn.dropout(lstm_output_flat, 1-self.params.drop_out_rate)
                tf.add_to_collection('lstm_output_embeddings',
                                     _lstm_output_unpacked)

                inputs["lstm_outputs" + ("_reverse" if lstm_num==1 else "")] = lstm_output_flat
        return inputs

    def __call__(self, inputs, *args, **kwargs):
        return super(LanguageModel, self).__call__(inputs, *args, **kwargs)

def language_model(inputs, params: user_params, is_predict:bool=False, trainable=True, name=None, dtype=None,
               activity_regularizer=None, **kwargs):
    layer = LanguageModel(params, is_predict, trainable, name, dtype,
                       activity_regularizer, **kwargs)
    return layer(inputs)





