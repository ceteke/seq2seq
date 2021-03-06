from model.seq2seq import Seq2Seq
import tensorflow as tf

sess = tf.Session()
# python3 -u  train_s2s.py -e 10 -i 100 -u 512 -g 5.0 -n 2 -em 500 -l 1.0 -d 0.3 -b 32 -o output/ -s 2000
s2s = Seq2Seq(sess, 512, [10], [10], 2, tensorboard_id=1, cell_type='GRU', mode='train', learning_rate=0.1, dropout=0.3, gradient_clip=5.0)
vars = tf.trainable_variables()
for v in vars:
    print(v)

print(sess.run(s2s.emb_test))
#s2s.restore('/Users/cem/Desktop/output/seq2seq')


