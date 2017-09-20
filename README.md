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
from seq2seq import Seq2Seq
 
sess = tf.Session()
s2s = Seq2Seq(sess=sess,hidden_units=1024, vocab_size=20, num_layers=1, embedding_size=200)
 
...
 
loss = s2s.train(encoder_inputs, encoder_input_lens, decoder_inputs, decoder_inputs_lens)
```

```encoder_inputs: [batch_size, encoder_seq_len]```  
```encoder_input_lens: [batch_size]```  
```decoder_inputs: [batch_size, decoder_seq_len]```  
```decoder_input_lens: [batch_size]```  
```loss:``` Perplexity

### TODO
- Attention
- Different embedding matrices for encoder and decoder
- Bidirectional encoder
- Beam search