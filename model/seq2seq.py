import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd()) # Import from current path
from .encoder import BasicEncoder
from .decoder import TrainingDecoder, InferenceDecoder
from .tf_utils import get_initializer

class Seq2Seq:
    def __init__(self, sess, hidden_units, vocab_sizes, embedding_sizes, num_layers, tensorboard_id=None, cell_type='LSTM',
                 attn=None, mode='train', learning_rate=0.001, dropout=None, gradient_clip=None, max_decode_len=None):
        '''
        :param sess: Tensoflow session
        :param hidden_units: Hidden units
        :param vocab_sizes: If has length 1 encoder and decoder share the same embedding, else first element is vocab size of
        encoder second element is vocab size of decoder.
        :param embedding_sizes:
        :param num_layers:
        :param cell_type:
        :param attn: Attention type. Can be 'bahdanau' or 'luong'
        :param mode: Can be 'train' or 'inference'. Dropouts and decoder is adjusted accordingly
        :param learning_rate: Learning rate
        :param dropout: Dropout between layers. This is applied to outputs of layers and the same dropout applied in encoder and decoder
        :param gradient_clip: Gradint clip. Clip by norm.
        :param max_decode_len: This must be given when mode='inference' max length decoder generates.
        '''

        assert (mode!='inference' or max_decode_len is not None), "At inference time max_decode_len must be given"

        self.sess = sess
        self.embedding_sizes = embedding_sizes
        self.vocab_sizes = vocab_sizes
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.mode = mode
        self.num_layers = num_layers
        self.dropout = dropout
        self.gradient_clip = gradient_clip
        self.max_decode_len = max_decode_len
        self.cell_type = cell_type
        self.variable_scope = 'seq2seq'
        self.attn = attn
        self.tensorboard_id = tensorboard_id

        assert (len(vocab_sizes) == len(embedding_sizes)), "Vocab sizes and embedding sizes length must be equal"
        assert (self.mode in ['inference', 'train']), "mode can be either 'inference' or 'train'"

        if len(vocab_sizes) == 1:
            vocab_size = self.vocab_sizes[0]
            embedding_size = self.embedding_sizes[0]

            with tf.variable_scope(self.variable_scope, initializer=get_initializer(tf.float32)):
                embedding = tf.get_variable(name='embedding', shape=[vocab_size, embedding_size],
                                                 dtype=tf.float32)

            if mode == 'train':
                self.encoder = BasicEncoder(self.cell_type, self.hidden_units, self.num_layers, self.dropout, embedding)
                self.decoder = TrainingDecoder(self.cell_type, self.hidden_units, self.num_layers, self.dropout, vocab_size,
                                               embedding, attn=self.attn)
            else:
                self.encoder = BasicEncoder(self.cell_type, self.hidden_units, self.num_layers, None, embedding)
                self.decoder = InferenceDecoder(self.cell_type, self.hidden_units, self.num_layers, self.max_decode_len,
                                                vocab_size, embedding, attn=self.attn)

        else:
            enc_vocab_size = self.vocab_sizes[0]
            enc_embedding_size = self.embedding_sizes[0]
            dec_vocab_size = self.vocab_sizes[1]
            dec_embedding_size = self.embedding_sizes[1]

            if self.mode == 'train':
                self.encoder = BasicEncoder(self.cell_type, self.hidden_units, self.num_layers, self.dropout, None,
                                            enc_vocab_size, enc_embedding_size)
                self.decoder = TrainingDecoder(self.cell_type, self.hidden_units, self.num_layers, self.dropout,
                                               dec_vocab_size, None, dec_embedding_size, attn=self.attn)
            elif self.mode == 'inference':
                self.encoder = BasicEncoder(self.cell_type, self.hidden_units, self.num_layers, None, None,
                                            enc_vocab_size, enc_embedding_size)
                self.decoder = InferenceDecoder(self.cell_type, self.hidden_units, self.num_layers, self.max_decode_len,
                                                dec_vocab_size, None, dec_embedding_size, attn=self.attn)

        self.init_variables()
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def init_variables(self):
        with tf.variable_scope('seq2seq'):
            self.global_step = tf.Variable(0, trainable=False)

    def build_graph(self):
        encoder_output, encoder_state = self.encoder.forward()
        # Decoder outputs loss if training, ids if prediction
        model_out = self.decoder.forward(encoder_outputs=encoder_output,encoder_states=encoder_state,
                                         encoder_sequence_lens=self.encoder.encoder_sequence_lens)
        if self.mode == 'train':
            self.loss = model_out
            tf.summary.scalar('loss', self.loss)
            self.train_op = self.init_optimizer(self.loss)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('tensorboard/{}'.format(self.tensorboard_id), self.sess.graph)
        elif self.mode == 'inference':
            self.predict_op = model_out

    def init_optimizer(self, loss):
        print("Building SGD Optimizer:\n\tlearning rate: {}\n\tgradient clip: {}".format(self.learning_rate,
                                                                                              self.gradient_clip),
              flush=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.gradient_clip is not None:
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
            train_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)
        else:
            train_op = self.optimizer.minimize(loss, global_step=self.global_step)

        return train_op

    def save(self, path, ow=True):
        saver = tf.train.Saver()
        if ow:
            save_path = saver.save(self.sess, save_path=path)
        else:
            save_path = saver.save(self.sess, save_path=path, global_step=self.global_step)

        print('model saved at {}'.format(save_path), flush=True)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path=path)

    def train(self, encoder_inputs, encoder_inputs_lens, decoder_inputs, decoder_inputs_lens):
        assert (self.mode == 'train'), "Can only train in 'train' mode"
        outputs = self.sess.run([self.train_op, self.loss, self.global_step, self.merged], feed_dict={self.encoder.encoder_inputs.name: encoder_inputs,
                                                                                        self.encoder.encoder_sequence_lens.name: encoder_inputs_lens,
                                                                                        self.decoder.decoder_inputs.name: decoder_inputs,
                                                                                        self.decoder.decoder_sequence_lens.name: decoder_inputs_lens})
        self.train_writer.add_summary(outputs[3], outputs[2])
        return outputs[1], outputs[2]

    def evaluate(self, encoder_inputs, encoder_inputs_lens, decoder_inputs, decoder_inputs_lens):
        assert (self.mode == 'train'), "Can only evaluate in 'train' mode"
        return self.sess.run(self.loss, feed_dict={self.encoder.encoder_inputs.name: encoder_inputs,
                                                   self.encoder.encoder_sequence_lens.name: encoder_inputs_lens,
                                                   self.decoder.decoder_inputs.name: decoder_inputs,
                                                   self.decoder.decoder_sequence_lens.name: decoder_inputs_lens})

    def predict(self, encoder_inputs, encoder_inputs_lens):
        assert (self.mode == 'inference'), "Can only predict in 'inference' mode"
        return self.sess.run(self.predict_op,
                             feed_dict={self.encoder.encoder_inputs.name: encoder_inputs,
                                        self.encoder.encoder_sequence_lens.name: encoder_inputs_lens})