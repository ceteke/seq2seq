import tensorflow as tf
from tensorflow.python.layers.core import Dense
import abc
from .tf_utils import get_multi_layer_rnn

class BaseDecoder(object):
    def __init__(self, cell_type, hidden_units, num_layers, vocab_size, embedding_size, embedding, eos_token):
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.eos_token = eos_token
        self.vocab_size = vocab_size
        self.variable_scope = 'decoder'
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.embedding = embedding

        if self.embedding is not None:
            self.embedding = embedding
        else:
            with tf.variable_scope(self.variable_scope):
                self.embedding = tf.get_variable(name='embedding', shape=[self.vocab_size, self.embedding_size],
                                                 dtype=tf.float32)

    @abc.abstractmethod
    def forward(self, encoder_states, encoder_sequence_lens):
        raise NotImplementedError

    @abc.abstractmethod
    def init_variables(self):
        raise NotImplementedError


class TrainingDecoder(BaseDecoder):
    def __init__(self, cell_type, hidden_units, num_layers, dropout, vocab_size, embedding, embedding_size=None, eos_token=1):
        super().__init__(cell_type, hidden_units, num_layers, vocab_size, embedding_size, embedding, eos_token)
        self.dropout = dropout
        self.init_variables()

    def init_variables(self):
        with tf.variable_scope(self.variable_scope):
            self.decoder_sequence_lens = tf.placeholder(tf.int32, shape=(None,), name='sequence_lengths')
            self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='input_sequences')

            self.decoder_multi_layer_cell = get_multi_layer_rnn(self.cell_type, self.hidden_units, self.num_layers, self.dropout)
            self.decoder_output_layer = Dense(self.vocab_size, name='output_projection')

    def forward(self, encoder_states, encoder_sequence_lens):
        batch_size = tf.shape(encoder_states[0][0])[0]
        with tf.variable_scope(self.variable_scope):
            eos_step = tf.ones(shape=[batch_size, 1], dtype=tf.int32) * self.eos_token

            decoder_train_inputs = tf.concat([eos_step, self.decoder_inputs], axis=1)
            decoder_train_targets = tf.concat([self.decoder_inputs, eos_step], axis=1)

            decoder_sequence_lens_train = self.decoder_sequence_lens + 1

            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_train_inputs)

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                sequence_length=decoder_sequence_lens_train,
                                                                name='training_helper')

            basic_train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_multi_layer_cell,
                                                                  helper=training_helper,
                                                                  initial_state=encoder_states,
                                                                  output_layer=self.decoder_output_layer)

            max_len = tf.reduce_max(decoder_sequence_lens_train)

            decoder_outputs_train, decoder_states_train, decoder_output_lens_train = tf.contrib.seq2seq.dynamic_decode(
                decoder=basic_train_decoder,
                impute_finished=True,
                maximum_iterations=max_len)

            decoder_output_logits = tf.identity(decoder_outputs_train.rnn_output)

            sequence_masks = tf.sequence_mask(lengths=decoder_sequence_lens_train, maxlen=max_len, dtype=tf.float32,
                                              name='masks')

            loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_output_logits,
                                                    targets=decoder_train_targets,
                                                    weights=sequence_masks)

            return loss


class InferenceDecoder(BaseDecoder):
    # TODO: Beam search
    def __init__(self, cell_type, hidden_units, num_layers, max_decode_len, vocab_size, embedding, embedding_size=None,
                 eos_token=1, beam_size=None):
        super().__init__(cell_type, hidden_units, num_layers, vocab_size, embedding_size, embedding, eos_token)
        self.beam_size = beam_size
        self.max_decode_len = max_decode_len
        self.init_variables()

    def init_variables(self):
        with tf.variable_scope(self.variable_scope):
            self.decoder_multi_layer_cell = get_multi_layer_rnn(self.cell_type, self.hidden_units, self.num_layers)
            self.decoder_output_layer = Dense(self.vocab_size, name='output_projection')

    def forward(self, encoder_states, encoder_sequence_lens):
        with tf.variable_scope(self.variable_scope):
            batch_size = tf.shape(encoder_states[0][0])[0]
            start_step = tf.ones([batch_size, ], dtype=tf.int32) * self.eos_token
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=lambda idx: tf.nn.embedding_lookup(self.embedding, idx),
                start_tokens=start_step,
                end_token=self.eos_token)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_multi_layer_cell,
                                                                helper=inference_helper,
                                                                initial_state=encoder_states,
                                                                output_layer=self.decoder_output_layer)
            decoder_outputs, decoder_states, decoder_output_lens = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder,
                maximum_iterations=self.max_decode_len)

            return decoder_outputs.sample_id
