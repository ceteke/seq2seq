import tensorflow as tf

class Encoder:
    def __init__(self, hidden_units, num_layers, dropout):
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout

        self.init_variables()

    def init_variables(self):
        '''
        Initializes LSTM layers of the encoder. Initial paramters are uniformly initialized between -0.1 and 0.1
        :return: None
        '''
        with tf.variable_scope("encoder", initializer=tf.random_uniform_initializer(-0.1, 0.1)):
            self.encoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='input_sequences')
            self.encoder_sequence_lens = tf.placeholder(tf.int32, shape=(None,), name='sequence_lengths')

            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_units)
            if self.dropout is not None:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=(1.0 - self.dropout),
                                                     output_keep_prob=(1.0 - self.dropout))
            self.encoder_multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)

    def forward(self, embedding, time_major=False):
        encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
        encoder_outputs, encoder_final_states = tf.nn.dynamic_rnn(cell=self.encoder_multi_layer_cell,
                                                                  inputs=encoder_inputs_embedded,
                                                                  sequence_length=self.encoder_sequence_lens,
                                                                  time_major=time_major,
                                                                  dtype=tf.float32)

        return encoder_outputs, encoder_final_states