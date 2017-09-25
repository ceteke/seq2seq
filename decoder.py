import tensorflow as tf
from tensorflow.python.layers.core import Dense

class Decoder:
    def __init__(self, hidden_units, num_layers, dropout, max_decode_len, vocab_size, mode='train', eos_token=1):
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_decode_len = max_decode_len
        self.mode = mode
        self.eos_token = eos_token
        self.vocab_size = vocab_size

        self.init_variables()

    def init_variables(self):
        with tf.variable_scope("decoder"):
            if self.mode == 'train':
                self.decoder_sequence_lens = tf.placeholder(tf.int32, shape=(None,), name='sequence_lengths')
                self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='input_sequences')

            cells = []
            for _ in range(self.num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_units)
                if self.dropout is not None and self.mode=='train':
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0 - self.dropout)
                cells.append(cell)

            self.decoder_multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            self.decoder_output_layer = Dense(self.vocab_size, name='output_projection')

    def forward(self, encoder_states, encoder_sequence_lens, embedding, vocab_size):
        batch_size = tf.shape(encoder_states[0][0])[0]

        with tf.variable_scope('decoder', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
            if self.mode == 'train':
                eos_step = tf.ones(shape=[batch_size, 1], dtype=tf.int32) * self.eos_token

                decoder_train_inputs = tf.concat([eos_step, self.decoder_inputs], axis=1)
                decoder_train_targets = tf.concat([self.decoder_inputs, eos_step], axis=1)

                decoder_sequence_lens_train = self.decoder_sequence_lens + 1

                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_train_inputs)

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

            elif self.mode == 'inference':
                start_step = tf.ones([batch_size,], dtype=tf.int32) * self.eos_token
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=lambda idx: tf.nn.embedding_lookup(embedding, idx),
                                                                            start_tokens=start_step,
                                                                            end_token=self.eos_token)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_multi_layer_cell,
                                                                    helper=inference_helper,
                                                                    initial_state=encoder_states,
                                                                    output_layer=self.decoder_output_layer)
                decoder_outputs, decoder_states, decoder_output_lens = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                                                         maximum_iterations=self.max_decode_len)

                return decoder_outputs.sample_id