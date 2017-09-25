from seq2seq import Seq2Seq
import tensorflow as tf

sess = tf.Session()

s2s = Seq2Seq(sess=sess,hidden_units=1024, vocab_size=20, num_layers=1, embedding_size=200, mode='inference', max_decode_len=20)
