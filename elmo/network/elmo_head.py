import tensorflow as tf
from tensorflow.python.estimator.canned import prediction_keys, metric_keys
from tensorflow.python.estimator.canned.head import _Head, _classification_output, _DEFAULT_SERVING_KEY, \
    _PREDICT_SERVING_KEY, LossSpec, _summary_key
from tensorflow.python.estimator.export import export_output
import numpy as np
from elmo.parameters import user_params


class ELMoHEAD(_Head):
    def __init__(self, params: user_params, name ="ELMoHEAD"):
        self.params = params
        self._name = name

    @property
    def name(self):
        return self._name

    def create_estimator_spec(
            self, features, mode, logits, labels=None, optimizer=None,
            train_op_fn=None, regularization_losses=None):
        """Returns an `EstimatorSpec`.

        Args:
          features: Input `dict` of `Tensor` or `SparseTensor` objects.
          mode: Estimator's `ModeKeys`.
          logits: a tensor array with 2 element, one for start positon probilities
          , and one for the end postion probilities.
          the shape is `[batch_size, logits_dimension]`.
          labels: a tensor with demention [batch_size, 2], 2 position for true position in a doc
          optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
            Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
            updates variables and increments `global_step`.
          train_op_fn: Function that takes a scalar loss `Tensor` and returns
            `train_op`. Used if `optimizer` is `None`.
          regularization_losses: A list of additional scalar losses to be added to
            the training loss, such as regularization losses. These losses are
            usually expressed as a batch average, so for best results users need to
            set `loss_reduction=SUM_OVER_BATCH_SIZE` or
            `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
            avoid scaling errors.
        Returns:
          `EstimatorSpec`.
        Raises:
          ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
            mode, or if both are set.
        """
        with tf.name_scope(self._name, 'head'):
            # Predict.
            pred_keys = prediction_keys.PredictionKeys
            predictions = {}

            if mode == tf.estimator.ModeKeys.PREDICT:
                output = export_output.PredictOutput(predictions)
                return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.PREDICT,
                    predictions=predictions,
                    export_outputs={
                        _DEFAULT_SERVING_KEY: output,
                        _PREDICT_SERVING_KEY: output
                    })


            training_loss, unreduced_loss, weights, label_ids = self.create_loss(
                features=features, mode=mode, logits=logits, labels=labels)

            if regularization_losses:
                regularization_loss = tf.add_n(regularization_losses)
                regularized_training_loss = tf.add_n(
                    [training_loss, regularization_loss])
            else:
                regularization_loss = None
                regularized_training_loss = training_loss

            # Eval.
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.EVAL,
                    predictions=predictions,
                    loss=regularized_training_loss,
                    eval_metric_ops=self._eval_metric_ops(
                        labels=label_ids,
                        predict=None, #tf.sign(tf.abs(tf.reshape(only_max,[d1,d2*d3*d3]))),
                        location =None, # loc,
                        pro = max,
                        unreduced_loss=unreduced_loss,
                        regularization_loss=regularization_loss))

            # Train.
            if optimizer is not None:
                if train_op_fn is not None:
                    raise ValueError('train_op_fn and optimizer cannot both be set.')
                train_op = optimizer.minimize(
                    regularized_training_loss,
                    global_step=tf.train.get_global_step())
            elif train_op_fn is not None:
                train_op = train_op_fn(regularized_training_loss)
            else:
                raise ValueError('train_op_fn and optimizer cannot both be None.')
        with tf.name_scope(''):
            keys = metric_keys.MetricKeys
            tf.summary.scalar(
                _summary_key(self._name, keys.LOSS),
                regularized_training_loss)

            if regularization_loss is not None:
                tf.summary.scalar(
                    _summary_key(self._name, keys.LOSS_REGULARIZATION),
                    regularization_loss)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            predictions=predictions,
            loss=regularized_training_loss,
            train_op=train_op)

    def create_loss(self, features, mode, logits, labels):
        softmax_dim = self.params.projection_dim
        softmax_init = tf.random_normal_initializer(0.0,
                                                    1.0 / np.sqrt(softmax_dim))

        softmax_W = tf.get_variable(
            'W', [self.params.n_tokens_vocab, softmax_dim],
            dtype=tf.float32,
            initializer=softmax_init
        )
        softmax_b = tf.get_variable(
            'b', [self.params.n_tokens_vocab],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))


        individual_losses = []
        zipped = [(features["lstm_outputs"], labels["next_token_id"]), (features["lstm_outputs_reverse"], labels["next_token_id_reverse"])]
        for lstm_output_flat, next_token  in zipped:
            next_token_id_flat = tf.reshape(next_token, [-1, 1])

            with tf.control_dependencies([lstm_output_flat]):
                losses = tf.nn.sampled_softmax_loss(
                    softmax_W, softmax_b,
                    next_token_id_flat, lstm_output_flat,
                    self.params.n_negative_samples_batch,
                    self.params.n_tokens_vocab,
                    num_true=1)

            losses = tf.expand_dims(losses, 1)
            # individual_losses.append(tf.reduce_mean(losses))
            individual_losses.append(losses)

        # return  0.5 * (individual_losses[0] + individual_losses[1])
        unreduced_loss = 0.5 * (individual_losses[0] + individual_losses[1])
        loss = tf.reduce_mean(unreduced_loss)
        return LossSpec(
            training_loss=loss,
            unreduced_loss=unreduced_loss,
            weights=None,
            processed_labels=labels)

    def _eval_metric_ops(
            self, labels,predict, location,pro,unreduced_loss, regularization_loss):

        return {}