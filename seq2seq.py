import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq:
    def __init__(self, sess, hidden_units, vocab_size, num_layers, embedding_size, mode='train', learning_rate=0.001,
                 dropout=None, gradient_clip=None, max_decode_len=None, is_single_embedding=True):

        assert (mode!='inference' or max_decode_len is not None), "At inference time max_decode_len must be given"

        self.sess = sess
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.mode = mode
        self.num_layers = num_layers
        self.dropout = dropout
        self.gradient_clip = gradient_clip
        self.max_decode_len = max_decode_len
        self.is_single_embedding = is_single_embedding

        self.encoder = Encoder(hidden_units=self.hidden_units, num_layers=self.num_layers, dropout=self.dropout)

        self.decoder = Decoder(hidden_units=self.hidden_units, num_layers=self.num_layers, dropout=self.dropout,
                               max_decode_len=self.max_decode_len, mode=self.mode, vocab_size=self.vocab_size)

        self.init_variables()
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def init_variables(self):
        with tf.variable_scope('seq2seq', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
            self.global_step = tf.Variable(0, trainable=False)
            self.embedding = tf.get_variable(name='embedding', shape=[self.vocab_size, self.embedding_size],
                                             dtype=tf.float32)

    def build_graph(self):
        print("Building Seq2Seq:\n\t# Layers:{}\n\t# Hidden Units:{}\n\tdropout: {}\n\tembedding_size: {}".format(
            self.num_layers, self.hidden_units, self.dropout,self.embedding_size)
              , flush=True)
        encoder_output, encoder_state = self.encoder.forward(embedding=self.embedding)
        # Decoder outputs loss if training, ids if prediction
        model_out = self.decoder.forward(encoder_states=encoder_state,
                                         encoder_sequence_lens=self.encoder.encoder_sequence_lens,
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding)
        if self.mode == 'train':
            self.loss = model_out
            self.train_op = self.init_optimizer(self.loss)
        elif self.mode == 'inference':
            self.predict_op = model_out

    def init_optimizer(self, loss):
        print("Building ADAM Optimizer:\n\tlearning rate: {}\n\tgradient clip: {}".format(self.learning_rate,
                                                                                              self.gradient_clip),
              flush=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.gradient_clip is not None:
            trainable_variables = tf.trainable_variables()
            gradients = tf.gradients(loss, trainable_variables)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
            train_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_variables), global_step=self.global_step)
        else:
            train_op = self.optimizer.minimize(loss, global_step=self.global_step)

        return train_op

    def save(self, path):
        saver = tf.train.Saver()

        save_path = saver.save(self.sess, save_path=path, global_step=self.global_step)
        print('model saved at {}'.format(save_path), flush=True)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path=path)

    def train(self, encoder_inputs, encoder_inputs_lens, decoder_inputs, decoder_inputs_lens):
        outputs = self.sess.run([self.train_op, self.loss, self.global_step], feed_dict={self.encoder.encoder_inputs.name: encoder_inputs,
                                                                                        self.encoder.encoder_sequence_lens.name: encoder_inputs_lens,
                                                                                        self.decoder.decoder_inputs.name: decoder_inputs,
                                                                                        self.decoder.decoder_sequence_lens.name: decoder_inputs_lens})
        return outputs[1], outputs[2]

    def evaluate(self, encoder_inputs, encoder_inputs_lens, decoder_inputs, decoder_inputs_lens):
        return self.sess.run(self.loss, feed_dict={self.encoder.encoder_inputs.name: encoder_inputs,
                                                   self.encoder.encoder_sequence_lens.name: encoder_inputs_lens,
                                                   self.decoder.decoder_inputs.name: decoder_inputs,
                                                   self.decoder.decoder_sequence_lens.name: decoder_inputs_lens})

    def predict(self, encoder_inputs, encoder_inputs_lens):
        return self.sess.run(self.predict_op,
                             feed_dict={self.encoder.encoder_inputs.name: encoder_inputs,
                                        self.encoder.encoder_sequence_lens.name: encoder_inputs_lens})