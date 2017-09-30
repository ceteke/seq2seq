# Seq2Seq in Tensorflow
To run this code you have to have dependencies installed in ``requirements.txt``.  

Prerequisites:
- A Python 3.6 environment

To install dependencies to your environment simply run 
 
``source /path/to/your/environment/bin/activate``  
``pip install -r requirements.txt``  

If you have GPU, please see Tensorflow GPU installation.
#### How to use?
```python
import tensorflow as tf
from model.seq2seq import Seq2Seq
 
sess = tf.Session()
s2s = Seq2Seq(sess=sess,hidden_units=1024, vocab_sizes=[2000,3000], num_layers=1, embedding_sizes=[200, 300])
 
...
 
loss, global_step = s2s.train(encoder_inputs, encoder_input_lens, decoder_inputs, decoder_inputs_lens)
```

* If ```vocab_sizes``` and ```embedding_sizes``` parameters has length of 1, encoder and decoder uses same
embedding matrix. If it has length of 2 encoder uses ```vocab_sizes[0]```, ```embedding_sizes[0]``` and decoder
uses ```vocab_sizes[1]```, ```embedding_sizes[1]```.
* Same ```dropout``` is applied to each layer of encoder and decoder.
* ```attn``` argument can be 'luong' or 'bahdanau'

```encoder_inputs: [batch_size, encoder_seq_len]```  
```encoder_input_lens: [batch_size]```  
```decoder_inputs: [batch_size, decoder_seq_len]```  
```decoder_input_lens: [batch_size]```  
```loss:``` Perplexity

### TODO
- Bidirectional encoder
- Beam search

#### References
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
https://www.tensorflow.org/tutorials/seq2seq#tensorflow_seq2seq_library