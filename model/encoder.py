import tensorflow as tf
from .tf_utils import get_multi_layer_rnn, get_initializer
import abc
import logging

class BaseEncoder(object):
    def __init__(self, cell_type, hidden_units, num_layers, vocab_size, embedding_size, embedding):
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.variable_scope = 'encoder'
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = embedding

        if self.embedding is not None:
            self.embedding = embedding
        else:
            with tf.variable_scope(self.variable_scope, initializer=get_initializer(tf.float32)):
                self.embedding = tf.get_variable(name='embedding', shape=[self.vocab_size, self.embedding_size],
                                                 dtype=tf.float32)

        with tf.variable_scope(self.variable_scope):
            self.encoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='input_sequences')
            self.encoder_sequence_lens = tf.placeholder(tf.int32, shape=(None,), name='sequence_lengths')


    @abc.abstractmethod
    def init_variables(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError

class BasicEncoder(BaseEncoder):
    def __init__(self, cell_type, hidden_units, num_layers, dropout, embedding, vocab_size=None, embedding_size=None):
        super().__init__(cell_type, hidden_units, num_layers, vocab_size, embedding_size, embedding)
        self.dropout = dropout

        self.init_variables()

    def init_variables(self):
        with tf.variable_scope(self.variable_scope, initializer=get_initializer(tf.float32)):
            self.encoder_multi_layer_cell = get_multi_layer_rnn(self.cell_type, self.hidden_units, self.num_layers, self.dropout)

    def forward(self):
        with tf.variable_scope(self.variable_scope, initializer=get_initializer(tf.float32)):
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            encoder_outputs, encoder_final_states = tf.nn.dynamic_rnn(cell=self.encoder_multi_layer_cell,
                                                                      inputs=encoder_inputs_embedded,
                                                                      sequence_length=self.encoder_sequence_lens,
                                                                      time_major=False,
                                                                      dtype=tf.float32)

            return encoder_outputs, encoder_final_states