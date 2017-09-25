from seq2seq import Seq2Seq
import tensorflow as tf

sess = tf.Session()
# python3 -u  train_s2s.py -e 100 -i 100 -u 128 -n 1 -em 300 -l 0.01 -d 0.5 -b 32 -o output/ -s 2000 > log.txt &
s2s = Seq2Seq(sess=sess,hidden_units=1024, vocab_size=50003, num_layers=1, embedding_size=300, mode='inference', max_decode_len=20)
s2s.restore('/Users/cem/Desktop/output/seq2seq')


