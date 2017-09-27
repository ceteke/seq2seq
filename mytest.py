from model.seq2seq import Seq2Seq
import tensorflow as tf

sess = tf.Session()
# python3 -u  train_s2s.py -e 10 -i 100 -u 512 -g 5.0 -n 2 -em 500 -l 1.0 -d 0.3 -b 32 -o output/ -s 2000
s2s = Seq2Seq(sess=sess,hidden_units=1024, vocab_size=50003, num_layers=1, embedding_size=300, mode='train', max_decode_len=20, gradient_clip=5.0)
#s2s.restore('/Users/cem/Desktop/output/seq2seq')


